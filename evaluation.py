import argparse
import logging
import random
from pathlib import Path
import pandas as pd
import pickle
import itertools

import numpy as np
import torch
from torch.nn import functional as F
import torchvision
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

import evaluation
import utils
from evaluation import compute_emotional_alignment
from data import DataLoader, ImageField, EmotionField, TextField, RawField, Merge
from models import Captioner
from models import clip
from models.blip import blip_feature_extractor
from data import ArtEmis, ArtEmisDetectionsField
from six.moves import cPickle, range

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
_logger = logging.getLogger('train')
_print_freq = 50
device = torch.device('cuda')
torch.backends.cudnn.benchmark = True


def load_vocabulary(file_name):
    with open(file_name, 'rb') as f:
        vocab = pickle.load(f)
    return vocab


def unpickle_data(file_name, python2_to_3=False):
    """ Restore data previously saved with pickle_data().
    :param file_name: file holding the pickled data.
    :param python2_to_3: (boolean), if True, pickle happened under python2x, unpickling under python3x.
    :return: an generator over the un-pickled items.
    Note, about implementing the python2_to_3 see
        https://stackoverflow.com/questions/28218466/unpickling-a-python-2-object-with-python-3
    """
    in_file = open(file_name, 'rb')
    if python2_to_3:
        size = cPickle.load(in_file, encoding='latin1')
    else:
        size = cPickle.load(in_file)

    for _ in range(size):
        if python2_to_3:
            yield cPickle.load(in_file, encoding='latin1')
        else:
            yield cPickle.load(in_file)
    in_file.close()
    
    
def flatten(l):
    if isinstance(l, list):
        for v in l:
            yield from flatten(v)
    else:
        yield l

        
def evaluate_emoalign(ground_truth, captions, split):
    wiki_art_img_dir = '/dir'
    references_file = '/dir'
    text2emo_path = '/dir'
    text2emo_bert_path = '/dir'
    vocab_path = '/dir'
    gt_data = next(unpickle_data(references_file))

    train_utters = gt_data['train']['references_pre_vocab']
    train_utters = list(itertools.chain(*train_utters))
    print('Training Utterances', len(train_utters))
    unique_train_utters = set(train_utters)
    print('Unique Training Utterances', len(unique_train_utters))
    
    gt_data = gt_data[split]
    print('Images Captioned', len(gt_data))
    for data in gt_data:
        gt_data['path']= "/dir" + '/' + gt_data['art_style'] + '/' + gt_data ['painting'] + '.jpg'
    
    txt2emo_clf = torch.load(text2emo_path, map_location=device)
    txt2emo_vocab = load_vocabulary(vocab_path)
    print('vocab size', len(txt2emo_vocab))

    merged = pd.merge(gt_data, captions)
    hypothesis = merged.caption # i.e., use references that do not have <UNK>
    ref_emotions = merged.emotion
    emoalign_score = compute_emotional_alignment(hypothesis, ref_emotions, txt2emo_clf, txt2emo_vocab, device=device)
    
    _logger.info('EMOTIONAL ALIGNMENT SCORE: ')
    _logger.info(emoalign_score)
    return emoalign_score


def evaluate_metrics(model, dataloader, text_field, emotion_encoder):
    model.eval()
    image_model.eval()    
    if emotion_encoder is not None:
        emotion_encoder.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    paths = {}
    gen = {}
    gts = {}
    captions = {'path':[], 'caption':[]}
    ground_truth = {'path':[], 'gtcaption':[]}
    gen_caps = pd.DataFrame(captions, columns=['path', 'caption'])
    gt_caps = pd.DataFrame(captions, columns=['path', 'gtcaption'])
    header = 'Evaluation metrics:'
    with torch.no_grad():
        if args.feature_extractor == 'CLIP' or args.feature_extractor == 'BLIP':
            for it, ((image_paths,images), captions_emotions) in enumerate(iter(metric_logger.log_every(dataloader, _print_freq, header))):
                images = images.to(device)
                images = image_model(images)  
                caps_gt, emotions = captions_emotions
                if emotion_encoder is not None:
                    emotions = torch.stack([torch.mode(emotion).values for emotion in emotions])
                    emotions = F.one_hot(emotions, num_classes=9)
                    emotions = emotions.type(torch.FloatTensor)
                    emotions = emotions.to(device)
                    enc_emotions = emotion_encoder(emotions)
                    enc_emotions = enc_emotions.unsqueeze(1).repeat(1, images.shape[1], 1)
                    images = torch.cat([images, enc_emotions], dim=-1)
                text, _ = model.beam_search(images, beam_size=5, out_size=1)
                caps_gen = text_field.decode(text)          

                for i, (path_i, gts_i, gen_i) in enumerate(zip(image_paths, caps_gt, caps_gen)):
                    paths['%d_%d' % (it, i)] = path_i
                    gen['%d_%d' % (it, i)] = [gen_i, ]
                    gts['%d_%d' % (it, i)] = gts_i
                    _logger.info(path_i)
                    _logger.info(gen_i)
                    gen_caps = gen_caps.append({'path': path_i, 'caption': gen_i}, ignore_index=True)
                    gt_caps = gt_caps.append({'path': path_i, 'gtcaption': gts_i}, ignore_index=True)
                    
        elif args.feature_extractor == 'ResNet101':
            for it, (detections, captions_emotions) in enumerate(iter(metric_logger.log_every(dataloader, _print_freq, header))):
                detections = detections.to(device)
                caps_gt, emotions = captions_emotions
                if emotion_encoder is not None:
                    emotions = torch.stack([torch.mode(emotion).values for emotion in emotions])
                    emotions = F.one_hot(emotions, num_classes=9)
                    emotions = emotions.type(torch.FloatTensor)
                    emotions = emotions.to(device)
                    enc_emotions = emotion_encoder(emotions)
                    enc_emotions = enc_emotions.unsqueeze(1).repeat(1, detections.shape[1], 1)
                    detections = torch.cat([detections, enc_emotions], dim=-1)
                text, _ = model.beam_search(detections, beam_size=5, out_size=1)
                caps_gen = text_field.decode(text)          

                for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                    gen['%d_%d' % (it, i)] = [gen_i, ]
                    gts['%d_%d' % (it, i)] = gts_i
                    gen_caps = gen_caps.append({'path': gts_i, 'caption': gen_i}, ignore_index=True)
                    gt_caps = gt_caps.append({'path': gen_i, 'gtcaption': gts_i}, ignore_index=True)
               
    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)

    #samples = [(paths[k], gen[k][0]) for k in list(paths.keys())[:20]]
    scores, _ = evaluation.compute_all_scores(gts, gen)
    #_logger.info('GEN_CAPS:')
    #_logger.info(gen_caps)
    #_logger.info(gt_caps)
    return scores, gt_caps, gen_caps


if __name__ == '__main__':
    _logger.info('CaMEL Evaluation')

    # Argument parsing
    parser = argparse.ArgumentParser(description='CaMEL Evaluation')
    parser.add_argument('--batch_size', type=int, default=25)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--annotation_folder', type=str, required=True)
    parser.add_argument('--image_folder', type=str, required=True)
    parser.add_argument('--saved_model_path', type=str, required=True)

    parser.add_argument('--clip_variant', type=str, default='RN50x16')
    parser.add_argument('--network', type=str, choices=('online', 'target'), default='target')
    parser.add_argument('--feature_extractor', type=str, choices=('CLIP', 'ResNet101', 'BLIP'), default='CLIP')
    parser.add_argument('--disable_mesh', action='store_true')

    parser.add_argument('--N_dec', type=int, default=3)
    parser.add_argument('--N_enc', type=int, default=3)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--m', type=int, default=40)
    parser.add_argument('--features_path', type=str)
    parser.add_argument('--use_emotion_labels', type=bool, default=False)
    parser.add_argument('--head', type=int, default=8)
    parser.add_argument('--with_pe', action='store_true')
    args = parser.parse_args()

    _logger.info(args)

    # Pipeline for image regions
    if args.feature_extractor == 'CLIP':
        clip_model, transform = clip.load(args.clip_variant, jit=False)
        image_model = clip_model.visual
        image_model.forward = image_model.intermediate_features
        image_field = ImageField(transform=transform)
    
    if args.feature_extractor == 'BLIP':
        blip_model = blip_feature_extractor(pretrained='/dir/blip_model_large.pth', image_size=224, vit='large')
        image_model = blip_model
        vision_width_large = 1024
        image_size = 224
        transform = transforms.Compose([
            transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ])
        image_field = ImageField(transform=transform)
        
    if args.feature_extractor == 'ResNet101':
        clip_model, transform = clip.load(args.clip_variant, jit=False)
        image_model = clip_model.visual
        image_model.forward = image_model.intermediate_features
        image_field = ArtEmisDetectionsField(detections_path=args.features_path, max_detections=50)

    # Pipeline for text
    text_field = TextField()
    emotions = [
        'amusement', 'awe', 'contentment', 'excitement', 
        'anger', 'disgust', 'fear', 'sadness', 'something else'
        ]
    emotion_field = EmotionField(emotions=emotions)
    
    emotion_dim = 0
    emotion_encoder = None
    if args.use_emotion_labels:
        emotion_dim = 10
        emotion_encoder = torch.nn.Sequential(
            torch.nn.Linear(9, emotion_dim)
            )
        emotion_encoder.to(device)
        
    args.d_ff = args.d_ff + emotion_dim
    if args.feature_extractor == 'BLIP':
        args.image_dim = 1024 + emotion_dim
    if args.feature_extractor == 'CLIP':
        args.image_dim = image_model.embed_dim + emotion_dim
    if args.feature_extractor == 'ResNet101':
        args.image_dim = 2048 + emotion_dim

    # Create the dataset and samplers
    dataset = ArtEmis(image_field, text_field, emotion_field, args.annotation_folder, args.feature_extractor)
    _, dataset_val, dataset_test = dataset.splits
    if args.feature_extractor == 'CLIP' or args.feature_extractor == 'BLIP':
        dict_dataset_val = dataset_val.image_dictionary({'image': Merge(RawField(), image_field), 'text': RawField(), 'emotion': emotion_field})#AAAAAAA
        dict_dataset_test = dataset_test.image_dictionary({'image': Merge(RawField(), image_field), 'text': RawField(), 'emotion': emotion_field})
    if args.feature_extractor == 'ResNet101':
        dict_dataset_val = dataset_val.image_dictionary({'image': image_field, 'text': RawField(), 'emotion': emotion_field})#AAAAAAA
        dict_dataset_test = dataset_test.image_dictionary({'image': image_field, 'text': RawField(), 'emotion': emotion_field})
    dict_dataloader_val = DataLoader(dict_dataset_val, batch_size=args.batch_size, num_workers=args.workers)
    dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=args.batch_size, num_workers=args.workers)

    # Create the model
    model = Captioner(args, text_field).to(device)
    model.forward = model.beam_search
    image_model = image_model.to(device)

    # Load the model weights
    fname = Path(args.saved_model_path)
    if not fname or not fname.is_file():
        raise ValueError(f'Model not found in {fname}')

    data = torch.load(fname, map_location=device)
    if args.network == 'target':
        _logger.info('Loading target network weights')
        model.load_state_dict(data['state_dict_t'])
    else:  # args.network == 'online'
        _logger.info('Loading online network weights')
        model.load_state_dict(data['state_dict_o'])
    
    if emotion_encoder is not None:
        _logger.info('Loading emotion encoder')
        emotion_encoder.to(device)
        emotion_encoder.load_state_dict(data['emotion_encoder'])
        
    # Validation captions
    _logger.info('Validation set')
    val_scores, val_gt_caps, val_gen_caps = evaluate_metrics(model, dict_dataloader_val, text_field, emotion_encoder)
    _logger.info(f'Validation scores {val_scores}')
    val_emo_score = evaluate_emoalign(val_gt_caps, val_gen_caps, 'val')
    _logger.info('Validation Emo-Align Score:')
    _logger.info(val_emo_score)

    # Test captions
    _logger.info('Test set')
    test_scores, test_gt_caps, test_gen_caps = evaluate_metrics(model, dict_dataloader_test, text_field, emotion_encoder)#AAAA
    _logger.info(f'Test scores {test_scores}')
    test_emo_score = evaluate_emoalign(test_gt_caps, test_gen_caps, 'test')
    _logger.info('Test Emo-Align Score:')
