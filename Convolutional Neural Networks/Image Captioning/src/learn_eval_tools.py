import logging
import torch.optim
import torch.utils.data
from torch.nn.utils import clip_grad_norm_


logger = logging.getLogger()


def train(train_loader, model, criterion, optimizer, grad_clip_c, epoch, print_freq, writer=None):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param model: Captioning model
    :param criterion: Loss layer
    :param optimizer: Optimizer to update model weights
    :param grad_clip_c: Value at which to clip the gradient norm
    :param batch_size: Batch size
    :param epoch: Epoch number
    :param print_freq: Frequency of loss printing
    """
    model.train()  # train mode (dropout and batchnorm is used)

    # Batches
    for i, (imgs, caps, caplens, img_paths) in enumerate(train_loader):
        # Forward prop.
        logits, decode_dict = model(imgs, caps, teacher_forcing_ratio=1)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps[:, 1:]

        # Calculate loss
        loss = criterion(torch.transpose(logits, 1, 2), targets)

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        clip_grad_norm_(model.parameters(), grad_clip_c)

        # Update weights
        optimizer.step()

        # Print status
        if i % print_freq == 0:
            report_str = "Training Epoch: {0} ({1}/{2})\tLoss: {3}".format(epoch, i, len(train_loader), loss)
            logger.info(report_str)
            writer.write(report_str + "\n")
            writer.flush()


def evaluate(eval_loader, model, criterion, evaluator, word_map, results_path, phase, writer=None):
    """
    Performs one epoch's validation.

    :param eval_loader: DataLoader for data
    :param model: Captioning model
    :param evaluator: Object to compute scoring metrics
    :param word_map: Dictionary mapping of token to index
    :param writer: Log file writer
    :return: Score of validation metrics, dictionary of all scores, and generated text-ground truth pair
    """
    model.eval()  # eval mode (no dropout or batchnorm)

    results = []
    with torch.no_grad():
        loss = 0.
        for i, (imgs, caps, caplens, allcaps, img_paths) in enumerate(eval_loader):
            # Forward prop.
            logits, decode_dict = model(imgs, caps, teacher_forcing_ratio=0)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps[:, 1:]

            loss += criterion(torch.transpose(logits, 1, 2), targets)

            if decode_dict[model.KEY_ATTN_SCORE] is not None:
                attn_scores = decode_dict[model.KEY_ATTN_SCORE].cpu().detach()
            else:
                attn_scores = None

            results.append(
                (
                    img_paths,
                    decode_dict[model.KEY_SEQ].cpu().detach(),
                    allcaps.cpu().detach(),
                    attn_scores
                )
            )

        report_str = "Total {} evaluation loss: {} (avg: {})".format(phase, loss, loss / len(eval_loader))
        logger.info(report_str)
        if writer:
            writer.write(report_str + "\n")

        (
            metric_score, all_scores
        ) = evaluator.decode_evaluate(
            results,
            {j: tok for tok, j in word_map.items()},
            results_path,
            writer=writer,
            verbosity=1
        )

    return metric_score, all_scores
