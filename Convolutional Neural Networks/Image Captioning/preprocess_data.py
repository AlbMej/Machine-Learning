import logging
from argparse import ArgumentParser

from src.datasets import create_input_files

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()


argparser = ArgumentParser()

argparser.add_argument("--dataset_type", default="coco", choices=["coco", "flickr8k", "flickr30k"],
                       help="Type of dataset.")
argparser.add_argument("--json_split_path", help="Path to caption json file.")
argparser.add_argument("--image_dir",
                       help="Directory containing all the images. " + \
                            "This should be the directory containing val2014, NOT val2014 itself.")
argparser.add_argument("--data_dir", help="Path to the directory to which the preprocessed data will be saved.")
argparser.add_argument("--caps_per_img", type=int, default=2, help="Number of captions per image.")
argparser.add_argument("--min_word_freq", type=int, default=2,
                       help="Minimum number of appearances in the training data of a particular word. "
                            + "Words below this threshold will be replaced with the UNK token.")
argparser.add_argument("--max_len", type=int, default=50, help="Maximum length of any caption.")
argparser.add_argument("--use_all_train", default=False, action="store_true",
                       help="Whether or not to use the full training data.")
argparser.add_argument("--train_percentage", type=float, default=0.1,
                       help="Percentage of training data that should be processed.")
argparser.add_argument("--val_percentage", type=float, default=0.1,
                       help="Percentage of validation data that should be processed.")
argparser.add_argument("--test_percentage", type=float, default=0.1,
                       help="Percentage of testing data that should be processed.")

args = argparser.parse_args()


create_input_files(
    dataset=args.dataset_type,
    split_json_path=args.json_split_path,
    image_folder=args.image_dir,
    captions_per_image=args.caps_per_img,
    min_word_freq=args.min_word_freq,
    output_folder=args.data_dir,
    max_len=args.max_len,
    use_all_train=args.use_all_train,
    train_percentage=args.train_percentage,
    val_percentage=args.val_percentage,
    test_percentage=args.test_percentage
)
