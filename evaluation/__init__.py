from .bleu import Bleu
from .meteor import Meteor
from .rouge import Rouge
from .cider import Cider
from .spice import Spice
from .emotion_alignment import emotional_alignment, compute_emotional_alignment
from .tokenizer import PTBTokenizer


def compute_all_scores(gts, gen):
    metrics = (Bleu(), Meteor(), Rouge(), Cider(), Spice())
    all_score = {}
    all_scores = {}
    for metric in metrics:
        score, scores = metric.compute_score(gts, gen)
        all_score[str(metric)] = score
        all_scores[str(metric)] = scores

    return all_score, all_scores
