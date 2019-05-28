import os
import numpy as np
import h5py
import json
import logging
import torch
from torch.utils.data import Dataset
from scipy.misc import imread, imresize
from collections import Counter
from random import choice, sample

import src.constants as C


logger = logging.getLogger()


class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, data_name, split, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Open hdf5 file where images are stored
        self.h = h5py.File(os.path.join(data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')
        self.imgs = self.h['images']

        # Captions per image
        self.cpi = self.h.attrs['captions_per_image']

        # Load encoded captions (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)

        # Load image paths (completely into memory)
        with open(os.path.join(data_folder, split + '_IMAGEPATHS_' + data_name + '.json'), 'r') as j:
            self.img_paths = json.load(j)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.captions)

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img = torch.cuda.FloatTensor(self.imgs[i // self.cpi] / 255.)
        if self.transform is not None:
            img = self.transform(img)

        caption = self.captions[i]
        caplen = self.caplens[i]
        img_path = self.img_paths[i // self.cpi]

        if self.split is 'TRAIN':
            return img, caption, caplen, img_path
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find scores
            all_captions = \
                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)]
            return img, caption, caplen, all_captions, img_path

    def __len__(self):
        return self.dataset_size


class CaptionProcessor(object):
    """
       An object to be used as the collate_fn in a PyTorch DataLoader to create batches.
    """
    def __init__(self,
                 sort: bool = False,
                 gpu: bool = False,
                 padding_idx: int = C.PAD_INDEX):
        """
        :param sort: Whether or not to sort the captions by length
        :param gpu: Whether or not to use a GPU
        :param padding_idx: Index of padding token
        """
        self.sort = sort
        self.gpu = gpu
        self.padding_idx = padding_idx

    def process(self, batch: list):
        if self.sort:
            batch.sort(key=lambda x: x[2], reverse=True)

        batch_cols = tuple(zip(*batch))
        if len(batch_cols) == 4:
            batch_imgs, batch_captions, batch_caplens, batch_fnames = batch_cols
            batch_all_captions = None
        else:
            batch_imgs, batch_captions, batch_caplens, batch_all_captions, batch_fnames = batch_cols

        if len(batch_caplens) > 1:
            max_cap_len = max(*batch_caplens)
        else:
            max_cap_len = batch_caplens[0]
        batch_imgs = torch.stack(batch_imgs, dim=0)
        batch_captions = torch.cuda.LongTensor([curr_cap[:max_cap_len] for curr_cap in batch_captions])
        batch_caplens = torch.cuda.LongTensor(batch_caplens)
        if batch_all_captions:
            batch_all_captions = torch.cuda.LongTensor(batch_all_captions)

        if self.gpu:
            batch_imgs = batch_imgs.cuda()
            batch_captions = batch_captions.cuda()
            batch_caplens = batch_caplens.cuda()
            if batch_all_captions:
                batch_all_captions = batch_all_captions.cuda()

        if batch_all_captions is not None:
            return batch_imgs, batch_captions, batch_caplens, batch_all_captions, batch_fnames
        else:
            return batch_imgs, batch_captions, batch_caplens, batch_fnames


def create_input_files(dataset,
                       split_json_path,
                       image_folder,
                       captions_per_image,
                       min_word_freq,
                       output_folder,
                       max_len=100,
                       use_all_train=False,
                       train_percentage=1.0,
                       val_percentage=1.0,
                       test_percentage=1.0):
    """
    Creates input files for training, validation, and test data.

    :param dataset: name of dataset, one of 'coco', 'flickr8k', 'flickr30k'
    :param karpathy_json_path: path of Karpathy JSON file with splits and captions
    :param image_folder: folder with downloaded images
    :param captions_per_image: number of captions to sample per image
    :param min_word_freq: words occuring less frequently than this threshold are binned as <unk>s
    :param output_folder: folder to save files
    :param max_len: don't sample captions longer than this length
    :param use_all_train: whether or not to use all the training data
    :param train_percentage: percentage of training data that should be used
    """

    assert dataset in {'coco', 'flickr8k', 'flickr30k'}

    # Read split JSON
    with open(split_json_path, 'r') as j:
        data = json.load(j)

    # Read image paths and captions for each image
    train_image_paths = []
    train_image_captions = []
    val_image_paths = []
    val_image_captions = []
    test_image_paths = []
    test_image_captions = []
    word_freq = Counter()

    test_set_name = {"test"}
    val_set_name = {"val"}
    train_set_name = {"restval"}
    if use_all_train:
        train_set_name.add("train")

    all_set_names = test_set_name | val_set_name | train_set_name

    for img in data['images']:
        if img["split"] not in all_set_names:
            continue

        captions = []
        for c in img['sentences']:
            # Update word frequency
            word_freq.update(c['tokens'])
            if len(c['tokens']) <= max_len:
                captions.append(c['tokens'])

        if len(captions) == 0:
            continue

        path = os.path.join(image_folder, img['filepath'], img['filename']) if dataset == 'coco' else os.path.join(
            image_folder, img['filename'])

        if img['split'] in train_set_name:
            train_image_paths.append(path)
            train_image_captions.append(captions)
        elif img['split'] in val_set_name:
            val_image_paths.append(path)
            val_image_captions.append(captions)
        elif img['split'] in test_set_name:
            test_image_paths.append(path)
            test_image_captions.append(captions)

    # Sanity check
    assert len(train_image_paths) == len(train_image_captions)
    assert len(val_image_paths) == len(val_image_captions)
    assert len(test_image_paths) == len(test_image_captions)

    num_train_examples = int(train_percentage * len(train_image_paths))
    train_image_paths = train_image_paths[:num_train_examples]
    train_image_captions = train_image_captions[:num_train_examples]

    num_val_examples = int(val_percentage * len(val_image_paths))
    val_image_paths = val_image_paths[:num_val_examples]
    val_image_captions = val_image_captions[:num_val_examples]

    num_test_examples = int(test_percentage * len(test_image_paths))
    test_image_paths = test_image_paths[:num_test_examples]
    test_image_captions = test_image_captions[:num_test_examples]

    # Create word map
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + len(C.TOKEN_PADS) for v, k in enumerate(words)}
    word_map[C.UNK] = C.UNK_INDEX
    word_map[C.SOS] = C.SOS_INDEX
    word_map[C.EOS] = C.EOS_INDEX
    word_map[C.PAD] = C.PAD_INDEX

    # Create a base/root name for all output files
    base_filename = dataset + '_' + str(captions_per_image) + '_cap_per_img_' + str(min_word_freq) + '_min_word_freq'

    # Save word map to a JSON
    with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w') as j:
        json.dump(word_map, j)

    # Sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files
    for impaths, imcaps, split in [(train_image_paths, train_image_captions, 'TRAIN'),
                                   (val_image_paths, val_image_captions, 'VAL'),
                                   (test_image_paths, test_image_captions, 'TEST')]:

        with h5py.File(os.path.join(output_folder, split + '_IMAGES_' + base_filename + '.hdf5'), 'a') as h:
            # Make a note of the number of captions we are sampling per image
            h.attrs['captions_per_image'] = captions_per_image

            # Create dataset inside HDF5 file to store images
            images = h.create_dataset('images', (len(impaths), 3, 256, 256), dtype='uint8')

            logger.info("Reading %s images and captions, storing to file...\n".format(split))

            enc_captions = []
            caplens = []
            img_paths = []

            for i, path in enumerate(impaths):

                # Sample captions
                if len(imcaps[i]) < captions_per_image:
                    captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
                else:
                    captions = sample(imcaps[i], k=captions_per_image)

                # Sanity check
                assert len(captions) == captions_per_image

                # Read images
                img = imread(impaths[i])
                if len(img.shape) == 2:
                    img = img[:, :, np.newaxis]
                    img = np.concatenate([img, img, img], axis=2)
                img = imresize(img, (256, 256))
                img = img.transpose(2, 0, 1)
                assert img.shape == (3, 256, 256)
                assert np.max(img) <= 255

                # Save image to HDF5 file
                images[i] = img

                img_paths.append(path)

                for j, c in enumerate(captions):
                    # Encode captions
                    enc_c = [word_map[C.SOS]] + [word_map.get(word, word_map[C.UNK]) for word in c] + [
                        word_map[C.EOS]] + [word_map[C.PAD]] * (max_len - len(c))

                    # Find caption lengths
                    c_len = len(c) + 2

                    enc_captions.append(enc_c)
                    caplens.append(c_len)

            # Sanity check
            assert images.shape[0] * captions_per_image == len(enc_captions) == len(caplens)
            assert images.shape[0] == len(img_paths)

            # Save encoded captions and their lengths to JSON files
            with open(os.path.join(output_folder, split + '_CAPTIONS_' + base_filename + '.json'), 'w') as j:
                json.dump(enc_captions, j)

            with open(os.path.join(output_folder, split + '_CAPLENS_' + base_filename + '.json'), 'w') as j:
                json.dump(caplens, j)

            with open(os.path.join(output_folder, split + '_IMAGEPATHS_' + base_filename + '.json'), 'w') as j:
                json.dump(img_paths, j)

    logger.info("Saved input files with data_name: {}".format(base_filename))
