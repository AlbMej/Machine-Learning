import sys
import json
import pickle
import logging
import random
import re
import collections

import torch
import torch.nn as nn

import src.constants as C
sys.path.append('src/pycocoevalcap')
sys.path.append('src/pycocoevalcap/bleu')
sys.path.append('src/pycocoevalcap/rouge')
sys.path.append('src/pycocoevalcap/meteor')
sys.path.append('src/pycocoevalcap/cider')
from src.pycocoevalcap.bleu.bleu import Bleu
from src.pycocoevalcap.rouge.rouge import Rouge
from src.pycocoevalcap.meteor.meteor import Meteor
from src.pycocoevalcap.cider.cider import Cider

logger = logging.getLogger()


class CaptionEvaluator(object):
    """
    Object to translate sequences of indices to tokens as well as run evaluations.
    """
    def __init__(self, val_metric="Bleu_4"):
        """
        :param val_metric: Metric that should be used to select the best model
        """
        self.val_metric = val_metric
        self.bad_tokens = list(zip(*filter(lambda x: C.UNK not in x, C.TOKEN_PADS)))[0]
        self.scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
        ]

    def convert(self, data):
        if isinstance(data, str):
            return data.encode('utf-8')
        elif isinstance(data, collections.Mapping):
            return dict(map(self.convert, data.items()))
        elif isinstance(data, collections.Iterable):
            return type(data)(map(self.convert, data))
        else:
            return data

    def format_scores(self, score_dict):
        """
        Return string for preporting scores.
        :param score_dict: Dictionary of metric name to score pairs
        :return:
        """
        sort_scores = sorted(score_dict.items(), key=lambda x: x[0])
        score_val_str = ["Scores:"]
        for metric, score in sort_scores:
            score_val_str.append("\t{}: {} ({:.2f})".format(metric, score, score * 100.0))
        return "\n".join(score_val_str)

    def to_words(self, img_paths, seqs, attn_scores, idx2token_map):
        """
        Translate indices to tokens.
        :param img_paths: List of image filenames
        :param seqs: Batch of generated captions
        :param idx2token_map: Mapping from index to token
        :param attn_scores: Attention scores
        :return: Dictionary of image path to captions list pairs
        """
        result = {}
        result_attn = {}
        if seqs.dim() == 2:
            for j in range(seqs.size(0)):
                example = " ".join([idx2token_map[seqs[j, k].item()] for k in range(seqs[j].size(0))
                                    if idx2token_map[seqs[j, k].item()] not in self.bad_tokens])
                result[img_paths[j]] = [example]
                if attn_scores is not None:
                    result_attn[img_paths[j]] = attn_scores[j, :].tolist()
        elif seqs.dim() == 3:
            for j in range(seqs.size(0)):
                all_examples = []
                for k in range(seqs.size(1)):
                    example = " ".join([idx2token_map[seqs[j, k, l].item()] for l in range(seqs[j, k].size(0))
                                        if idx2token_map[seqs[j, k, l].item()] not in self.bad_tokens])
                    all_examples.append(example)
                result[img_paths[j]] = all_examples
                if attn_scores is not None:
                    result_attn[img_paths[j]] = attn_scores[j, :].tolist()

        return result, result_attn

    def score(self, ref, hypo):
        """
        Run scorers.
        :param ref: Reference captions
        :param hypo: Hypothesis captions
        :return: Dictionary of metric name to score pairs
        """
        final_scores = {}
        for scorer, method in self.scorers:
            score, scores = scorer.compute_score(ref, hypo)
            if type(score) == list:
                for m, s in zip(method, score):
                    final_scores[m] = s
            else:
                final_scores[method] = score

        return final_scores

    def evaluate(self, live=True, **kwargs):
        """
        Format candidate and reference captions and evaluate.
        :param live: Wether or not caption data should be loaded from a file or if they are given
        :param kwargs:
        :return: Dictionary of metric name to score pairs
        """
        verbosity = kwargs.pop("verbosity", 1)
        if live:
            temp_ref = kwargs.pop('ref', {})
            cand = kwargs.pop('cand', {})
        else:
            reference_path = kwargs.pop('ref', '')
            candidate_path = kwargs.pop('cand', '')

            # load caption data
            with open(reference_path, 'rb') as f:
                temp_ref = pickle.load(f)
            with open(candidate_path, 'rb') as f:
                cand = pickle.load(f)

        # make dictionary
        hypo = {}
        ref = {}
        for i, (vid, caption) in enumerate(cand.items()):
            hypo[i] = caption
            ref[i] = temp_ref[vid]

        # compute scores
        final_scores = self.score(ref, hypo)

        if verbosity > 0:
            logger.info(self.format_scores(final_scores))

        return final_scores

    def decode_evaluate(self, results, idx_token, results_path=None, writer=None, verbosity=1):
        """
        Function to perform decoding and evaluation.
        :param results: List of tuples (list of image file names, generated sequences, ground truth sequences)
        :param idx_token: Mapping from index to token
        :param results_path: Path to write out the results
        :param writer: Log file writer
        :param verbosity: Verbosity of evaluation
        :return:
        """
        cand = {}
        cand_attn = {}
        ref = {}
        for img_paths_b, preds_b, golds_b, attn_scores_b in results:
            cand_b, cand_attn_b = self.to_words(img_paths_b, preds_b, attn_scores_b, idx_token)
            cand.update(cand_b)
            cand_attn.update(cand_attn_b)

            ref_b, _ = self.to_words(img_paths_b, golds_b, None, idx_token)
            ref.update(ref_b)

        curr_scores = self.evaluate(live=True, cand=cand, ref=ref, verbosity=verbosity)

        if results_path:
            with open(results_path, 'w', encoding="utf-8") as res_f:
                json.dump(
                    {"scores": curr_scores, "candidates": cand, "references": ref, "attention": cand_attn},
                    res_f
                )

        if writer:
            writer.write(self.format_scores(curr_scores))
            writer.flush()

        return curr_scores[self.val_metric], curr_scores


def save_checkpoint(filename, epoch, epochs_since_improvement, model, optimizer, scores, is_best):
    """
    Saves model checkpoint.

    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param bleu4: validation BLEU-4 score for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    state = {
        "epoch": epoch,
        "epochs_since_improvement": epochs_since_improvement,
        "scores": scores,
        "model": model,
        "optimizer": optimizer
    }
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, filename + ".best")


def load_embeddings(path: str,
                    dimension: int,
                    vocab: dict = None,
                    skip_first_line: bool = True,
                    rand_range: list = None,
                    fine_tune_embeds: bool = True,
                    pad_index: int = 0
                    ):
    if not path:
        logger.info("No word embedding file given. Randomly initialized embeddings will be loaded.")
        return nn.Embedding(len(vocab), dimension, padding_idx=pad_index, sparse=False)

    logger.info('Scanning embedding file: {}'.format(path))

    embed_vocab = set()
    lower_mapping = {}  # lower case - original
    digit_mapping = {}  # lower case + replace digit with 0 - original
    digit_pattern = re.compile('\d')
    with open(path, 'r', encoding='utf-8') as r:
        if skip_first_line:
            r.readline()
        for line in r:
            try:
                token = line.split(' ')[0].strip()
                if token:
                    embed_vocab.add(token)
                    token_lower = token.lower()
                    token_digit = re.sub(digit_pattern, '0', token_lower)
                    if token_lower not in lower_mapping:
                        lower_mapping[token_lower] = token
                    if token_digit not in digit_mapping:
                        digit_mapping[token_digit] = token
            except UnicodeDecodeError:
                continue

    token_mapping = collections.defaultdict(list)  # embed token - vocab token
    for token in vocab:
        token_lower = token.lower()
        token_digit = re.sub(digit_pattern, '0', token_lower)
        if token in embed_vocab:
            token_mapping[token].append(token)
        elif token_lower in lower_mapping:
            token_mapping[lower_mapping[token_lower]].append(token)
        elif token_digit in digit_mapping:
            token_mapping[digit_mapping[token_digit]].append(token)

    logger.info('Loading embeddings')
    if rand_range is not None:
        rand_range.sort()
        weight = [[random.uniform(rand_range[0], rand_range[1]) for _ in range(dimension)] for _ in range(len(vocab))]
    else:
        weight = [[.0] * dimension for _ in range(len(vocab))]
    with open(path, 'r', encoding='utf-8') as r:
        if skip_first_line:
            r.readline()
        for line in r:
            try:
                segs = line.rstrip().split(' ')
                token = segs[0]
                if token in token_mapping:
                    vec = [float(v) for v in segs[1:]]
                    for t in token_mapping.get(token):
                        weight[vocab[t]] = vec.copy()
            except UnicodeDecodeError:
                continue
            except ValueError:
                continue
    embed = nn.Embedding(
        len(vocab),
        dimension,
        padding_idx=pad_index,
        sparse=False,
        _weight=torch.FloatTensor(weight)
    )
    if not fine_tune_embeds:
        embed.weight.requires_grad = False
    return embed


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
