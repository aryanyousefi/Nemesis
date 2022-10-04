"""
Measuring the emotion-alignment between a generation and the ground-truth (emotion).
The MIT License (MIT)
Originally created at 8/31/20, for Python 3.x
Copyright (c) 2021 Panos Achlioptas (pachlioptas@gmail.com) & Stanford Geometric Computing Lab
"""

import torch
from torch import nn
import numpy as np
import pandas as pd


def iterate_in_chunks(l, n):
    """Yield successive 'n'-sized chunks from iterable 'l'.
    Note: last chunk will be smaller than l if n doesn't divide l perfectly.
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]

@torch.no_grad()
def image_to_emotion(img2emo_clf, data_loader, device):
    """ For each image of the underlying dataset predict an emotion
    :param img2emo_clf: nn.Module
    :param data_loader: torch loader of dataset to iterate
    :param device: gpu placement
    :return:
    """
    img2emo_clf.eval()
    emo_of_img_preds = []
    for batch in data_loader:
        predictions = img2emo_clf(batch['image'].to(device)).cpu()
        emo_of_img_preds.append(predictions)
    emo_of_img_preds = torch.cat(emo_of_img_preds)
    return emo_of_img_preds


@torch.no_grad()
def text_to_emotion(txt2em_clf, encoded_tokens, device, batch_size=1000):
    """
    :param txt2em_clf:
    :param encoded_tokens: Tensor carrying the text encoded
    :param device:
    :param batch_size:
    :return:
    """
    txt2em_clf.eval()
    emotion_txt_preds = []
    for chunk in iterate_in_chunks(encoded_tokens, batch_size):
        emotion_txt_preds.append(txt2em_clf(chunk.to(device)).cpu())

    emotion_txt_preds = torch.cat(emotion_txt_preds)
    maximizers = torch.argmax(emotion_txt_preds, -1)
    return emotion_txt_preds, maximizers


def unique_maximizer(a_list):
    """ if there is an element of the input list that appears
    strictly more frequent than any other element
    :param a_list:
    :return:
    """
    u_elements, u_cnt = np.unique(a_list, return_counts=True)
    has_umax = sum(u_cnt == u_cnt.max()) == 1
    umax = u_elements[u_cnt.argmax()]
    return has_umax, umax


def dominant_maximizer(a_list):
    """ if there is an element of the input list that appears
    at least half the time
    :param a_list:
    :return:
    """
    u_elements, u_cnt = np.unique(a_list, return_counts=True)

    has_umax = u_cnt.max() >= len(a_list) / 2

    if len(u_cnt) >= 2: # make sure the second most frequent does not match the first.
        a, b = sorted(u_cnt)[-2:]
        if a == b:
            has_umax = False

    umax = u_elements[u_cnt.argmax()]
    return has_umax, umax


def cross_entropy(pred, soft_targets):
    """ pred: unscaled logits
        soft_targets: target-distributions (i.e., sum to 1)
    """
    logsoftmax = nn.LogSoftmax(dim=1)
    return torch.mean(torch.sum(-soft_targets * logsoftmax(pred), 1))


def occurrence_list_to_distribution(list_of_ints, n_support):
    """e.g., [0, 8, 8, 8] -> [1/4, 0, ..., 3/4, 0, ...]"""
    distribution = np.zeros(n_support, dtype=np.float32)
    for i in list_of_ints:
        distribution[i] += 1
    distribution /= sum(distribution)
    return distribution


def emotional_alignment(hypothesis, emotions, vocab, txt2em_clf, device):
    """ text 2 emotion, then compare with ground-truth.
    :param hypothesis:
    :param emotions: (list of list of int) human emotion-annotations (ground-truth) e.g., [[0, 1] [1]]
    :param vocab:
    :param txt2em_clf:
    :param device:
    :return:
    """

    # from text to emotion
    hypothesis_tokenized = hypothesis.apply(lambda x: x.split())
    max_len = hypothesis_tokenized.apply(lambda x: len(x)).max()
    hypothesis = hypothesis_tokenized.apply(lambda x: np.array(vocab.encode(x, max_len=max_len)))
    hypothesis = torch.from_numpy(np.vstack(hypothesis))
    pred_logits, pred_maximizer = text_to_emotion(txt2em_clf, hypothesis, device)

    # convert emotion lists to distributions to measure cross-entropy
    n_emotions = 9
    emo_dists = torch.from_numpy(np.vstack(emotions.apply(lambda x: occurrence_list_to_distribution(x, n_emotions))))
    x_entropy = cross_entropy(pred_logits, emo_dists).item()

    # constrain predictions to those of images with dominant maximizer of emotion
    has_max, maximizer = zip(*emotions.apply(dominant_maximizer))
    emotion_mask = np.array(has_max)
    masked_emotion = np.array(maximizer)[emotion_mask]

    guess_correct = masked_emotion == pred_maximizer[emotion_mask].cpu().numpy()
    accuracy = guess_correct.mean()

    return accuracy, x_entropy


def compute_emotional_alignment(hypothesis, ref_emotions, txt2emo_clf, text2emo_vocab, device='cuda'):
        stat_track = ['mean', 'std']
        results=[]
        emo_accuracy, emo_xentopy = emotional_alignment(hypothesis, ref_emotions, text2emo_vocab, txt2emo_clf, device)
        stats = pd.Series(emo_accuracy, dtype=float)
        stats = stats.describe()[stat_track]
        stats = pd.concat([pd.Series({'metric': 'Emo-Alignment-ACC'}), stats])
        results.append(stats)

        stats = pd.Series(emo_xentopy, dtype=float)
        stats = stats.describe()[stat_track]
        stats = pd.concat([pd.Series({'metric': 'Emo-Alignment-XENT'}), stats])
        results.append(stats)
        #print('EMO-ALIGN: done')
        return results