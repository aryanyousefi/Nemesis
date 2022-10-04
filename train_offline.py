import argparse
import collections
import itertools
import logging
import multiprocessing
import os
import random
import pickle
import pandas as pd
from pathlib import Path
from shutil import copyfile

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import NLLLoss
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from itertools import repeat

import evaluation
import utils
from data import DataLoader
from data import ArtEmis
from data import ImageField, TextField, EmotionField, ArtEmisDetectionsField, RawField, Merge
from evaluation import PTBTokenizer, Cider
from evaluation import compute_emotional_alignment
from models import Captioner
from models import clip
from six.moves import cPickle, range#BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB

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
    
    
def flatten(l):#AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
    if isinstance(l, list):#AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
        for v in l:#AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
            yield from flatten(v)#AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
    else:#AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
        yield l#AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA

        
def evaluate_emoalign(ground_truth, captions, split):#AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABBBBBBBBBBBBBBBBBBBBBBBB
    wiki_art_img_dir = '/home/ayousefi/wikiart'
    references_file = '/home/ayousefi/artemis-master/artemis/preprocessed_for_deep_net/artemis_gt_references_grouped.pkl'
    text2emo_path = '/home/ayousefi/artemis-master/artemis/classifiers/txt_to_emotion_lstm_based/best_model.pt'
    vocab_path = '/home/ayousefi/artemis-master/artemis/preprocessed_for_deep_net/vocabulary.pkl'    
    gt_data = next(unpickle_data(references_file))#BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB

    train_utters = gt_data['train']['references_pre_vocab']#BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB
    train_utters = list(itertools.chain(*train_utters))  #BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB
#    print('Training Utterances', len(train_utters))#BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB
    unique_train_utters = set(train_utters)#BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB
#    print('Unique Training Utterances', len(unique_train_utters))#BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB
    
    gt_data = gt_data[split]
#    print('Images Captioned', len(gt_data))
    for data in gt_data:
        gt_data['path']= "/home/ayousefi/wikiart" + '/' + gt_data['art_style'] + '/' + gt_data ['painting'] + '.jpg'
    
    txt2emo_clf = torch.load(text2emo_path, map_location=device)
    #_logger.info('GT_DATA: \n')
    #_logger.info(gt_data)
    #_logger.info('CAPTIONS: \n')
    #_logger.info(captions)
    txt2emo_vocab = load_vocabulary(vocab_path)
#    print('vocab size', len(txt2emo_vocab))

#    for caption in captions:  # you might have sampled under several sampling configurations
    merged = pd.merge(gt_data, captions)#BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB
        #merged = pd.merge(ground_truth, captions)  # this ensures proper order of captions to gt (via accessing merged.captions)
    hypothesis = merged.caption # i.e., use references that do not have <UNK>
    ref_emotions = merged.emotion
    #_logger.info("MERGED:")
    #_logger.info(merged)#BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB
    #_logger.info("HYPOTHESIS:")
    #_logger.info(hypothesis)#BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB
    #_logger.info(ref_emotions)#BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB
    emoalign_score = compute_emotional_alignment(hypothesis, ref_emotions, txt2emo_clf, txt2emo_vocab, device='cuda')
    if args.phase == 'xe':
        _logger.info('EMOTIONAL ALIGNMENT SCORE: ')
        _logger.info(emoalign_score)
    return emoalign_score


def evaluate_metrics(model, dataloader, text_field, emotion_encoder=None):
    model.eval()
    if image_model_without_ddp:
        image_model_without_ddp.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    paths = {}
    gen = {}
    gts = {}
    captions = {'path':[], 'caption':[]}#AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
    ground_truth = {'path':[], 'gtcaption':[]}#AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
    gen_caps = pd.DataFrame(captions, columns=['path', 'caption'])#AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
    gt_caps = pd.DataFrame(captions, columns=['path', 'gtcaption'])#AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA

    header = f'Evaluation metrics Epoch: [{e}]'
    with torch.no_grad():
        for it, (images, captions_emotions) in enumerate(iter(metric_logger.log_every(dataloader, _print_freq, header))):
            images = images.to(device)
            caps_gt, emotions = captions_emotions
            if emotion_encoder is not None:
                emotions = torch.stack([torch.mode(emotion).values for emotion in emotions]) # pick the most frequent emotion
                emotions = F.one_hot(emotions, num_classes=9)
                emotions = emotions.type(torch.FloatTensor)
                emotions = emotions.to(device)
                enc_emotions = emotion_encoder(emotions)
                enc_emotions = enc_emotions.unsqueeze(1).repeat(1, images.shape[1], 1)
                images = torch.cat([images, enc_emotions], dim=-1)
            #if image_model_without_ddp:
            #    images = image_model_without_ddp(images)
            text, _ = model.beam_search(images, beam_size=5, out_size=1)
            caps_gen = text_field.decode(text)
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen['%d_%d_%d' % (args.rank, it, i)] = [gen_i, ]
                gts['%d_%d_%d' % (args.rank, it, i)] = gts_i
                gen_caps = gen_caps.append({'path': gts_i, 'caption': gen_i}, ignore_index=True)#AAAWRONGWONRONORNEGONEOQKFOEKRFOQAEMKC
                gt_caps = gt_caps.append({'path': gen_i, 'gtcaption': gts_i}, ignore_index=True)#ANKSDVNOAWEOFKQWPOEKFOQPWKEFOPQWKEOQKWE

    if args.distributed:
        paths_all = [None for _ in range(args.world_size)]
        gts_all = [None for _ in range(args.world_size)]
        gen_all = [None for _ in range(args.world_size)]
        dist.barrier()
        dist.all_gather_object(paths_all, paths)
        dist.all_gather_object(gts_all, gts)
        dist.all_gather_object(gen_all, gen)
        paths = dict(collections.ChainMap(*paths_all))
        gts = dict(collections.ChainMap(*gts_all))
        gen = dict(collections.ChainMap(*gen_all))
    
    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_all_scores(gts, gen)

    return scores, gt_caps, gen_caps


def train_xe(target_model, online_model, dataloader, optim, text_field, emotion_encoder=None):
    # Training with cross-entropy
    online_model.train()
    target_model.train()
    if emotion_encoder is not None:
        emotion_encoder.train()
    if args.distributed:
        train_sampler.set_epoch(e)

    image_model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f'Training Epoch: [{e}]'

    for it, (detections, captions, emotions) in enumerate(metric_logger.log_every(dataloader, _print_freq, header)):
        detections, captions = detections.to(device), captions.to(device)
        if emotion_encoder is not None:
            emotions = F.one_hot(emotions, num_classes=9)
            emotions = emotions.type(torch.FloatTensor)
            emotions = emotions.to(device)
            enc_emotions = emotion_encoder(emotions)
            enc_emotions = enc_emotions.unsqueeze(1).repeat(1, detections.shape[1], 1)
            detections = torch.cat([detections, enc_emotions], dim=-1)

        online_logits = online_model(detections, captions)
        online_text = F.log_softmax(online_logits, dim=-1)
        online_text = online_text[:, :-1].contiguous()
        online_captions_gt = captions[:, 1:online_text.shape[1] + 1].contiguous()
        online_loss = loss_fn(online_text.view(-1, text_field._tokenizer.vocab_size), online_captions_gt.view(-1))

        loss = online_loss

        # Knowledge distillation
        with torch.no_grad():
            target_logits = target_model(detections, captions)

        distillation_loss = ((online_logits - target_logits) ** 2).mean()
        distillation_weight = args.distillation_weight
        distillation_loss *= distillation_weight

        metric_logger.update(distillation_loss=distillation_loss, online_loss=online_loss)
        loss += distillation_loss

        optim.zero_grad()
        loss.backward()
        optim.step()
        scheduler.step()

        # EMA update
        with torch.no_grad():
            params_s = list(online_model.parameters())
            params_t = list(target_model.parameters())
            if args.use_emotion_labels:
                params_s = list(online_model.parameters()) + list(emotion_encoder.parameters())
                params_t = list(target_model.parameters()) + list(emotion_encoder.parameters())
                
            torch._foreach_mul_(params_t, args.ema_weight)
            w = torch._foreach_mul(params_s, 1 - args.ema_weight)
            torch._foreach_add_(params_t, w)

        metric_logger.update(loss=loss)

    metric_logger.synchronize_between_processes()
    return metric_logger.loss.global_avg


def train_scst(target_model, online_model, dataloader, optim, cider, text_field, emotion_encoder=None):
    # Training with self-critical
    tokenizer_pool = multiprocessing.Pool(4)
    if args.distributed:
        seq_len = online_model.module.decoder.max_len
    else:
        seq_len = online_model.decoder.max_len
    beam_size = 5
    online_model.train()
    target_model.train()
    if args.distributed:
        train_sampler.set_epoch(e)

    image_model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Training Epoch: [{}]'.format(e)

    for it, (detections, captions_emotions) in enumerate(metric_logger.log_every(dataloader, _print_freq, header)):#XXXXXXXXXXXXXXXX
        detections = detections.to(device)
        caps_gt, emotions = captions_emotions
        if emotion_encoder is not None:
            emotions = torch.stack([torch.mode(emotion).values for emotion in emotions]) # pick the most frequent emotion
            emotions = F.one_hot(emotions, num_classes=9)
            emotions = emotions.type(torch.FloatTensor)
            emotions = emotions.to(device)
            enc_emotions = emotion_encoder(emotions)
            enc_emotions = enc_emotions.unsqueeze(1).repeat(1, detections.shape[1], 1)
            detections = torch.cat([detections, enc_emotions], dim=-1)
           
        online_outs, online_log_probs, online_logits = online_model(detections, beam_size=beam_size, out_size=beam_size,
                                                                    return_logits=True)

        # Rewards
        caps_gen = text_field.decode(online_outs.view(-1, seq_len))
        caps_gt = list(itertools.chain(*([c, ] * beam_size for c in caps_gt)))
        caps_gen, caps_gt = tokenizer_pool.map(evaluation.PTBTokenizer.tokenize, [caps_gen, caps_gt])
        reward = cider.compute_score(caps_gt, caps_gen)[1].astype(np.float32)
        reward = torch.from_numpy(reward).to(device).view(detections.shape[0], beam_size)
        reward_baseline = torch.mean(reward, -1, keepdim=True)

        avg_online_log_probs = torch.sum(online_log_probs, -1) / torch.sum(online_log_probs != 0, -1)
        reward_loss = -avg_online_log_probs * (reward - reward_baseline)
        reward_loss = reward_loss.mean()
        loss = reward_loss

        # Knowledge distillation
        with torch.no_grad():
            target_outs, target_log_probs, target_logits = target_model(detections, beam_size=beam_size, out_size=1,
                                                                        return_logits=True)

        best_target_logits = target_logits
        best_online_logits = online_logits[:, 0]

        mask = (best_online_logits == 0) | (best_target_logits == 0)
        best_online_logits.masked_fill_(mask, 0)
        best_target_logits.masked_fill_(mask, 0)

        distillation_loss = ((best_online_logits - best_target_logits) ** 2).mean()
        distillation_weight = args.distillation_weight
        distillation_loss *= distillation_weight

        metric_logger.update(distillation_loss=distillation_loss, reward_loss=reward_loss)
        loss += distillation_loss

        optim.zero_grad()
        loss.backward()
        optim.step()

        # EMA update
        with torch.no_grad():
            params_s = list(online_model.parameters())
            params_t = list(target_model.parameters())
            if args.use_emotion_labels:
                params_s = list(online_model.parameters()) + list(emotion_encoder.parameters())
                params_t = list(target_model.parameters()) + list(emotion_encoder.parameters())
            torch._foreach_mul_(params_t, args.ema_weight)
            w = torch._foreach_mul(params_s, 1 - args.ema_weight)
            torch._foreach_add_(params_t, w)

        metric_logger.update(loss=loss, reward=reward.mean())

    metric_logger.synchronize_between_processes()
    tokenizer_pool.close()
    return metric_logger.loss.global_avg, metric_logger.reward.global_avg


if __name__ == '__main__':
    _logger.info('CaMEL Training')

    # Argument parsing
    parser = argparse.ArgumentParser(description='CaMEL Training')
    parser.add_argument('--exp_name', type=str, default='camel')
    parser.add_argument('--batch_size', type=int, default=25)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--resume_last', action='store_true')
    parser.add_argument('--resume_best', action='store_true')
    parser.add_argument('--annotation_folder', type=str, required=True)
    parser.add_argument('--image_folder', type=str, required=True)

    parser.add_argument('--clip_variant', type=str, default='RN50x16')
    parser.add_argument('--distillation_weight', type=float, default=None)
    parser.add_argument('--ema_weight', type=float, default=0.999)
    parser.add_argument('--phase', type=str, default='xe', choices=('xe', 'scst'))
    parser.add_argument('--disable_mesh', action='store_true') #store_true originally
    parser.add_argument('--features_path', type=str)
    parser.add_argument('--N_dec', type=int, default=3)
    parser.add_argument('--N_enc', type=int, default=3)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--m', type=int, default=40)
    parser.add_argument('--head', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=10000)
    parser.add_argument('--with_pe', action='store_true')
    parser.add_argument('--use_emotion_labels', type=bool, default=False)

    args = parser.parse_args()

    if args.distillation_weight is None:
        args.distillation_weight = 0.1 if args.phase == 'xe' else 0.005

    args.exp_name += args.phase

    _logger.info(args)

    random.seed(1234)
    torch.manual_seed(1234)
    np.random.seed(1234)

    # Distributed initialization
    args.distributed = True #originally was False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    args.world_size = 1
    args.rank = 0  # global rank
    if args.distributed:
        torch.cuda.set_device('cuda:0')
        torch.distributed.init_process_group(backend='nccl', init_method='env://', rank=int(os.environ['SLURM_PROCID']))
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        _logger.info('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
                     % (args.rank, args.world_size))
        _logger.info('Process %d, total %d, running on node %s, CUDA_VISIBLE_DEVICES=%s.'
                     % (args.rank, args.world_size, os.uname()[1], os.environ['CUDA_VISIBLE_DEVICES']))
    else:
        _logger.info('Training with a single process on 1 GPUs.')
    assert args.rank >= 0

    # Pipeline for image regions
    clip_model, transform = clip.load(args.clip_variant, jit=False)
    image_model = clip_model.visual
    image_model.forward = image_model.intermediate_features
    image_field = ArtEmisDetectionsField(detections_path=args.features_path, max_detections=50)
    #args.image_dim = image_model.embed_dim

    # Pipeline for text
    text_field = TextField()
    
    emotions = [
        'amusement', 'awe', 'contentment', 'excitement', 
        'anger', 'disgust', 'fear', 'sadness', 'something else'
        ]
    emotion_field = EmotionField(emotions=emotions)

    # Create the dataset and samplers
    dataset = ArtEmis(image_field, text_field, emotion_field, args.annotation_folder, visual_encoder= 'ResNet101')
    dataset_train, dataset_val, dataset_test = dataset.splits
    dict_dataset_train = dataset_train.image_dictionary({'image': image_field, 'text': RawField(), 'emotion': emotion_field})
    dict_dataset_val = dataset_val.image_dictionary({'image': image_field, 'text': RawField(), 'emotion': emotion_field})
    dict_dataset_test = dataset_test.image_dictionary({'image': image_field, 'text': RawField(), 'emotion': emotion_field})
    dataset_scst = dataset_train.image_dictionary({'image': Merge(RawField(), image_field), 'text': RawField()})#XXXXXXXXXXXXXXXXXXXXXXXXXX
    #dataset_val = dataset_val.image_dictionary({'image': Merge(RawField(), image_field), 'text': RawField(), 'emotion': RawField()})
    #dataset_test = dataset_test.image_dictionary({'image': Merge(RawField(), image_field), 'text': RawField(), 'emotion': RawField()})
    
        # Model and dataloaders
    emotion_dim = 0
    emotion_encoder = None
    if args.use_emotion_labels:
        emotion_dim = 10
        emotion_encoder = torch.nn.Sequential(
            torch.nn.Linear(9, emotion_dim)
            )
        emotion_encoder.to(device)
    
    args.d_ff = args.d_ff + emotion_dim
    args.image_dim = 2048 + emotion_dim

    if args.phase == 'scst':
        ref_caps_train = list(i.text for i in dataset_train.examples)
        cider_train = Cider(PTBTokenizer.tokenize(ref_caps_train))
        dataset_train = dataset_train.image_dictionary({'image': image_field, 'text': RawField(), 'emotion': emotion_field})#YOU CAN CHANGE THIS INSTEAD OF 385

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
        val_sampler = torch.utils.data.distributed.DistributedSampler(dataset_val, shuffle=False)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    #else:
       # train_sampler = torch.utils.data.RandomSampler(dataset_train)
       # val_sampler = torch.utils.data.SequentialSampler(dataset_val)
       # test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    train_batch_size = args.batch_size if 'scst' not in args.phase else args.batch_size // 5
    #dataloader_train = DataLoader(dataset_train, batch_size=train_batch_size, num_workers=args.workers,
    #                              sampler=train_sampler, pin_memory=True)
    #dataloader_scst = DataLoader(dataset_scst, batch_size=train_batch_size, num_workers=args.workers,
    #                              sampler=train_sampler, pin_memory=True)#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    #dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size // 5, sampler=val_sampler, pin_memory=True)
    #dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size // 5, sampler=test_sampler,
    #                             pin_memory=True)
    dataloader_scst = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                      drop_last=True)
    dict_dataloader_train = DataLoader(dict_dataset_train, batch_size=args.batch_size // 5, shuffle=True,
                                           num_workers=args.workers)
    dict_dataloader_val = DataLoader(dict_dataset_val, batch_size=args.batch_size // 5)
    dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=args.batch_size // 5)                                                  
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                      drop_last=True)
    dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # Create the models
    target_model = Captioner(args, text_field).to(device)
    online_model = Captioner(args, text_field).to(device)
    online_model_without_ddp = online_model

    if args.phase == 'scst':
        target_model.forward = target_model.beam_search
        online_model.forward = online_model.beam_search

    image_model_without_ddp = None
    if image_model:
        image_model = image_model.to(device)
        image_model_without_ddp = image_model

    if args.distributed:
        online_model = DistributedDataParallel(online_model)
        if image_model:
            image_model = DistributedDataParallel(image_model)

    # Optimizers

    def lambda_lr(s):
        warm_up = args.warmup
        s += 1
        return (online_model_without_ddp.d_model ** -.5) * min(s ** -.5, s * warm_up ** -1.5)

    if args.phase == 'xe':
        base_lr = 1
    else:  # args.phase == 'scst'
        base_lr = 5e-6

    optim = Adam(online_model.parameters(), lr=base_lr, betas=(0.9, 0.98))
    if args.phase == 'xe':
        scheduler = LambdaLR(optim, lambda_lr)
    else:
        scheduler = None

    # Loss and variable initialization
    loss_fn = NLLLoss(ignore_index=0)
    best_val_cider = 0.0
    patience = 0
    start_epoch = 0

    # Load the model weights
    if args.resume_last or args.resume_best:
        if args.resume_last:
            fname = Path('/home/ayousefi/projects/def-kpassi/ayousefi/Epoch0_Grounded_Shuffled').joinpath(f'{args.exp_name}_last_emo.pth')
            #fname = Path('/home/ayousefi/camel/saved_models').joinpath(f'{args.exp_name}_last.pth')
        else:
            fname = Path('/home/ayousefi/camel/saved_models').joinpath(f'{args.exp_name}_best.pth')
        if fname.is_file():
            data = torch.load(fname)
            torch.set_rng_state(data['torch_rng_state'])
            torch.cuda.set_rng_state(data['cuda_rng_state'])
            np.random.set_state(data['numpy_rng_state'])
            random.setstate(data['random_rng_state'])

            target_model.load_state_dict(data['state_dict_t'])
            online_model_without_ddp.load_state_dict(data['state_dict_o'])
            emotion_encoder.load_state_dict(data['emotion_encoder'])

            optim.load_state_dict(data['optimizer'])
            if scheduler:
                scheduler.load_state_dict(data['scheduler'])
            start_epoch = data['epoch'] + 1
            best_val_cider = data['best_val_cider']
            val_cider = data['val_t_cider'] if 'val_t_cider' in data.keys() else data['val_o_cider']
            patience = data['patience']
            if patience >= 5:
                _logger.info(f'Loaded model with patience reached! Exit')
                exit()
            _logger.info(f'Resuming from epoch {data["epoch"]}, validation cider {val_cider}')
        else:
            _logger.warning(f'Saved model not found in "{fname.absolute()}"! Starting training from scratch')
    elif args.phase == 'scst':
        best_xe_model = args.exp_name.replace(args.phase, 'xe')
        #fname = Path('/home/ayousefi/camel/saved_models').joinpath(f'{best_xe_model}_best.pth')
        fname = Path('/home/ayousefi/projects/def-kpassi/ayousefi/Epoch9_Grounded_Shuffled').joinpath(f'{best_xe_model}_last_emo.pth')
        if not fname.is_file():
            raise FileNotFoundError(f'Saved model not found in "{fname.absolute()}"!')
        data = torch.load(fname)
        target_model.load_state_dict(data['state_dict_t'])
        online_model_without_ddp.load_state_dict(data['state_dict_o'])
        emotion_encoder.load_state_dict(data['emotion_encoder'])
        best_val_cider = data['best_val_cider']
        val_cider = data['val_t_cider'] if 'val_t_cider' in data.keys() else data['val_o_cider']
        test_cider = data['test_t_cider'] if 'test_t_cider' in data.keys() else data['test_o_cider']
        _logger.info(f'Resuming from XE epoch {data["epoch"]}, validation cider {val_cider}, test cider {test_cider}')

    # Training loop
    _logger.info("Training starts")

    for e in range(start_epoch, start_epoch + 100):
        if args.phase == 'xe':
            train_loss = train_xe(target_model, online_model, dataloader_train, optim, text_field, emotion_encoder)
        elif args.phase == 'scst':
            train_loss, reward = train_scst(target_model, online_model, dict_dataloader_train, optim,
                                            cider_train, text_field, emotion_encoder)

        # Online validation loss and scores
        scores,val_gt_caps, val_gen_caps = evaluate_metrics(online_model_without_ddp, dict_dataloader_val, text_field, emotion_encoder)
        _logger.info(f'Online validation scores {scores}')
        val_o_cider = scores['CIDEr']
        #val_emo_score = evaluate_emoalign(val_gt_caps, val_gen_caps, 'val')#AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA YOU CAN REMOVE VAL_GT_CAPS
        #_logger.info('Online validation Emo-Align score:')#AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
        #_logger.info(val_emo_score)#AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA

        # Target validation loss and scores
        scores, val_gt_caps, val_gen_caps = evaluate_metrics(target_model, dict_dataloader_val, text_field, emotion_encoder)
        _logger.info(f'Target validation scores {scores}')
        val_t_cider = scores['CIDEr']
        #val_emo_score = evaluate_emoalign(val_gt_caps, val_gen_caps, 'val')#AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA YOU CAN REMOVE VAL_GT_CAPS
        #_logger.info('Target validation Emo-Align score:')#AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
        #_logger.info(val_emo_score)#AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA

        # Online test scores
        scores, test_gt_caps, test_gen_caps = evaluate_metrics(online_model_without_ddp, dict_dataloader_test, text_field, emotion_encoder)
        _logger.info(f'Online test scores {scores}')
        test_o_cider = scores['CIDEr']
        #test_emo_score = evaluate_emoalign(test_gt_caps, test_gen_caps, 'test')#AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
        #_logger.info('Test Emo-Align Score:')#AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
        #_logger.info(test_emo_score)#AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
        

        # Target test scores
        scores, test_gt_caps, test_gen_caps = evaluate_metrics(target_model, dict_dataloader_test, text_field, emotion_encoder)
        _logger.info(f'Target test scores {scores}')
        test_t_cider = scores['CIDEr']

        # Prepare for next epoch
        best = False
        if val_t_cider >= best_val_cider:
            best_val_cider = val_t_cider
            patience = 0
            best = True
        else:
            patience += 1

        exit_train = False
        if patience == 5:
            _logger.info('patience reached.')
            exit_train = True

        if args.rank == 0:
            save_dict = {
                'torch_rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state(),
                'numpy_rng_state': np.random.get_state(),
                'random_rng_state': random.getstate(),
                'epoch': e,
                'emotion_encoder': emotion_encoder.state_dict(),
                'optimizer': optim.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler else None,
                'patience': patience,
                'best_val_cider': best_val_cider,
                'val_t_cider': val_t_cider,
                'test_t_cider': test_t_cider,
                'state_dict_t': target_model.state_dict(),
                'val_o_cider': val_o_cider,
                'test_o_cider': test_o_cider,
                'state_dict_o': online_model_without_ddp.state_dict()
            }

            if args.use_emotion_labels:
                torch.save(save_dict, f'/home/ayousefi/projects/def-kpassi/ayousefi/Epoch0_Grounded_Shuffled/{args.exp_name}_last_emo.pth')
            else:
                torch.save(save_dict, f'/home/ayousefi/projects/def-kpassi/ayousefi/Epoch0_Grounded_Shuffled/{args.exp_name}_last_emo.pth')

            if best:
                if args.use_emotion_labels:
                    copyfile(f'/home/ayousefi/projects/def-kpassi/ayousefi/Epoch0_Grounded_Shuffled/{args.exp_name}_last_emo.pth', f'/home/ayousefi/projects/def-kpassi/ayousefi/Epoch0_Grounded_Shuffled/{args.exp_name}_best_emo.pth')
                else:
                    copyfile(f'/home/ayousefi/projects/def-kpassi/ayousefi/Epoch0_Grounded_Shuffled/{args.exp_name}_last_emo.pth', f'/home/ayousefi/outputs/camel/saved_models/{args.exp_name}_best_emo.pth')

        if exit_train:
            break