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
from src.learn_eval_tools import evaluate
from src.utils import CaptionEvaluator


RANDOM_SEED = 6130
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

argparser = ArgumentParser()

argparser.add_argument('--data_dir', help="Path to the data directory.")
argparser.add_argument('--data_name', help="Name of data file based on preprocessing arguments.")
argparser.add_argument('--checkpoint_path', help='Path to pretrained model file.')
argparser.add_argument('--results', help="Path to results file.")
argparser.add_argument('--gpu', action='store_true')
argparser.add_argument('--device', default=0, type=int)
argparser.add_argument('--threads', default=1, type=int)


args = argparser.parse_args()

use_gpu = args.gpu and torch.cuda.is_available()
if use_gpu:
    torch.cuda.set_device(args.device)
else:
    args.device = "cpu"

# Results file
results_dir = args.results
assert results_dir and os.path.isdir(results_dir), 'Result dir is required'
results_file = os.path.join(results_dir, 'test.results.{}.json'.format(timestamp))

logger.info("Random seed: {}".format(RANDOM_SEED))
logger.info('----------')
logger.info('Parameters:')
for arg in vars(args):
    logger.info('\t{}: {}'.format(arg, getattr(args, arg)))
logger.info("Result file: {}".format(results_file))
logger.info('----------')
# Custom dataloaders
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
cap_proc = CaptionProcessor(sort=True, gpu=use_gpu, padding_idx=C.PAD_INDEX)
test_loader = torch.utils.data.DataLoader(
    CaptionDataset(args.data_dir, args.data_name, 'TEST', transform=transforms.Compose([normalize])),
    batch_size=1, shuffle=False, num_workers=args.threads, pin_memory=True, collate_fn=cap_proc.process)

evaluator = CaptionEvaluator()


# Read word map
word_map_file = os.path.join(args.data_dir, 'WORDMAP_' + args.data_name + '.json')
with open(word_map_file, 'r') as j:
    word_map = json.load(j)

# Load checkpoint
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

logger.info("Running testing")
curr_metric_score, all_scores = evaluate(
    eval_loader=test_loader,
    model=cap_model,
    criterion=criterion,
    evaluator=evaluator,
    word_map=word_map,
    results_path=results_file,
    phase="test",
    writer=None
)
logger.info("Finished Testing\n")
