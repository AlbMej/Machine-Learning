import os
import time
import json
import logging
import random
from argparse import ArgumentParser

import torch
import torchvision.transforms as transforms
from torch import nn

import src.constants as C
from src.datasets import CaptionDataset, CaptionProcessor
from src.layers import CNNEncoder, RNNDecoder, LSTMDecoder
from src.models import CaptioningModel
from src.learn_eval_tools import train, evaluate
from src.utils import CaptionEvaluator, adjust_learning_rate, save_checkpoint, load_embeddings


RANDOM_SEED = 6130
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

start_epoch = 0
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
best_metric_score = 0.
best_scores = {}

argparser = ArgumentParser()

argparser.add_argument('--data_dir', help="Path to the data directory.")
argparser.add_argument('--data_name', default="coco_2_cap_per_img_2_min_word_freq",help="Name of data file based on preprocessing arguments.")
argparser.add_argument('--log', help='Path to the log directory.')
argparser.add_argument('--model_path', help='Path to the directory to which the model file will be saved.')
argparser.add_argument('--checkpoint_path', help='Path to pretrained model file.')
argparser.add_argument('--results', help="Path to results file.")
argparser.add_argument('--batch_size', default=32, type=int, help='Batch size.')
argparser.add_argument('--max_epoch', default=10, type=int, help="Number of epochs.")
argparser.add_argument('--word_embed', help='Path to the pre-trained embedding file.')
argparser.add_argument('--fine_tune_encoder', default=False, action="store_true",
                       help="Wether or not to fine tune encoder CNN during training.")
argparser.add_argument('--fine_tune_embeds', default=False, action="store_true",
                       help="Wether or not to fine tune word embeddings during training.")
argparser.add_argument('--use_attention', default=False, action="store_true",
                       help="Wether or not to use an attention mechanism.")
argparser.add_argument('--word_embed_dim', type=int, default=50, help='Word embedding dimension.')
argparser.add_argument('--decoder_hidden_size', default=256, type=int,
                       help='Decoder hidden state size.')
argparser.add_argument('--decoder_cell_type', default="vanilla", choices=["vanilla", "lstm"],
                       help='Type of decoder cell.')
argparser.add_argument('--embed_dropout', default=0.6, type=float,
                       help='Word feature dropout probability.')
argparser.add_argument('--decoder_dropout', default=0.6, type=float,
                       help='Decoder output dropout probability.')
argparser.add_argument('--lr', default=0.001, type=float,
                       help='Encoder learning rate.')
argparser.add_argument('--grad_clipping', default=5.0, type=float)
argparser.add_argument('--gpu', action='store_true')
argparser.add_argument('--device', default=0, type=int)
argparser.add_argument('--threads', default=1, type=int)
argparser.add_argument("--val_metric", default="Bleu_4",
                       choices=["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr"],
                       help="Evaluation metric used to select the best model.")
argparser.add_argument("--print_freq", default=1, type=int,
                       help="Frequency of loss reporting during training.")


args = argparser.parse_args()

use_gpu = args.gpu and torch.cuda.is_available()
if use_gpu:
    torch.cuda.set_device(args.device)
else:
    args.device = "cpu"

# Model file
model_dir = args.model_path
assert model_dir and os.path.isdir(model_dir), 'Model output dir is required'
model_file = os.path.join(model_dir, 'model.{}.mdl'.format(timestamp))

# Results file
results_dir = args.results
assert results_dir and os.path.isdir(results_dir), 'Result dir is required'
results_file = os.path.join(results_dir, 'val.results.{}.json'.format(timestamp))

# Logging file
log_writer = None
if args.log:
    log_file = os.path.join(args.log, 'log.{}.txt'.format(timestamp))
    log_writer = open(log_file, 'a', encoding='utf-8')
    logger.addHandler(logging.FileHandler(log_file, encoding='utf-8'))
logger.info("Random seed: {}".format(RANDOM_SEED))
logger.info('----------')
logger.info('Parameters:')
for arg in vars(args):
    logger.info('\t{}: {}'.format(arg, getattr(args, arg)))
logger.info("\tModel full path: {}".format(model_file))

# Custom dataloaders
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
cap_proc = CaptionProcessor(sort=True, gpu=use_gpu, padding_idx=C.PAD_INDEX)
train_loader = torch.utils.data.DataLoader(
    CaptionDataset(args.data_dir, args.data_name, 'TRAIN', transform=transforms.Compose([normalize])),
    batch_size=args.batch_size, shuffle=True, num_workers=args.threads, pin_memory=True, collate_fn=cap_proc.process)
val_loader = torch.utils.data.DataLoader(
    CaptionDataset(args.data_dir, args.data_name, 'VAL', transform=transforms.Compose([normalize])),
    batch_size=args.batch_size, shuffle=False, num_workers=args.threads, pin_memory=True, collate_fn=cap_proc.process)

logger.info("\tTraining data size: {}".format(len(train_loader)))
logger.info("\tValidation data size: {}".format(len(val_loader)))

evaluator = CaptionEvaluator(val_metric=args.val_metric)
logger.info('----------')

# Read word map
word_map_file = os.path.join(args.data_dir, 'WORDMAP_' + args.data_name + '.json')
with open(word_map_file, 'r') as j:
    word_map = json.load(j)

# Initialize / load checkpoint
if not args.checkpoint_path:
    embed_layer = load_embeddings(
        args.word_embed,
        args.word_embed_dim,
        word_map,
        skip_first_line=True,
        rand_range=[-0.05, 0.05],
        fine_tune_embeds=args.fine_tune_embeds,
        pad_index=C.PAD_INDEX
    )

    encoder = CNNEncoder(
        proj_dim=args.decoder_hidden_size if not args.use_attention else None,
        feat_layer="conv" if args.use_attention else "fc",
        fine_tune=args.fine_tune_encoder
    )

    if args.decoder_cell_type == "vanilla":
        decoder = RNNDecoder(
            vocab_size=len(word_map),
            embed_size=args.word_embed_dim,
            encoder_state_size=encoder.output_size,
            hidden_size=args.decoder_hidden_size,
            decoder_out_dropout_prob=args.decoder_dropout,
            use_attention=args.use_attention
        )
    else:
        decoder = LSTMDecoder(
            vocab_size=len(word_map),
            embed_size=args.word_embed_dim,
            encoder_state_size=encoder.output_size,
            hidden_size=args.decoder_hidden_size,
            decoder_out_dropout_prob=args.decoder_dropout,
            use_attention=args.use_attention
        )

    cap_model = CaptioningModel(
        decoder_type=args.decoder_cell_type,
        sos_id=C.SOS_INDEX,
        eos_id=C.EOS_INDEX,
        word_embedding_layer=embed_layer,
        encoder=encoder,
        decoder=decoder,
        decoder_hidden_size=args.decoder_hidden_size,
        embed_dropout_prob=args.embed_dropout
    )

    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, cap_model.parameters()),
                                 lr=args.lr)
else:
    checkpoint = torch.load(args.checkpoint_path)
    start_epoch = checkpoint['epoch'] + 1
    epochs_since_improvement = checkpoint['epochs_since_improvement']
    best_scores = checkpoint['scores']
    cap_model = checkpoint["model"]
    optimizer = checkpoint['optimizer']

# Move to GPU, if available
if use_gpu:
    cap_model.cuda()

# Loss function
criterion = nn.CrossEntropyLoss(ignore_index=C.PAD_INDEX)
if use_gpu:
    criterion = criterion.cuda()

# Show model architecture
logger.info('----------')
logger.debug(cap_model)
logger.info('----------')

# Epochs
for epoch in range(start_epoch, args.max_epoch):

    # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
    if epochs_since_improvement == 20:
        break
    if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
        adjust_learning_rate(optimizer, 0.8)

    # One epoch's training
    logger.info("Running training for epoch {}".format(epoch))
    train(
        train_loader=train_loader,
        model=cap_model,
        criterion=criterion,
        optimizer=optimizer,
        grad_clip_c=args.grad_clipping,
        epoch=epoch,
        print_freq=args.print_freq,
        writer=log_writer
    )
    logger.info("Finished training epoch {}\n".format(epoch))

    # One epoch's validation
    logger.info("Running validation for epoch {}".format(epoch))
    curr_metric_score, all_scores = evaluate(
        eval_loader=val_loader,
        model=cap_model,
        criterion=criterion,
        evaluator=evaluator,
        word_map=word_map,
        results_path=results_file,
        phase="val",
        writer=log_writer
    )
    logger.info("Finished validation for epoch {}\n".format(epoch))

    # Check if there was an improvement
    is_best = curr_metric_score > best_metric_score
    best_metric_score = max(curr_metric_score, best_metric_score)
    if not is_best:
        epochs_since_improvement += 1
        logger.info("\nEpochs since last improvement: {}\n".format(epochs_since_improvement))
    else:
        best_scores = all_scores
        epochs_since_improvement = 0
        # Save checkpoint
    save_checkpoint(model_file, epoch, epochs_since_improvement, cap_model, optimizer, all_scores, is_best)

logger.info("Best model save file: {}".format(model_file))
logger.info("Best model scores: {}".format(best_scores))
