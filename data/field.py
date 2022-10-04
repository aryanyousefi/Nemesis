# coding: utf8
from itertools import takewhile

import torch
import pickle
import os
import base64
import sys
import csv
import numpy as np
from torch.utils.data.dataloader import default_collate
from torchvision.datasets.folder import default_loader
import warnings

from .tokenizer.simple_tokenizer import SimpleTokenizer as _Tokenizer

csv.field_size_limit(sys.maxsize)


class RawField(object):
    """ Defines a general datatype.

    Every dataset consists of one or more types of data. For instance,
    a machine translation dataset contains paired examples of text, while
    an image captioning dataset contains images and texts.
    Each of these types of data is represented by a RawField object.
    An RawField object does not assume any property of the data type and
    it holds parameters relating to how a datatype should be processed.

    Attributes:
        preprocessing: The Pipeline that will be applied to examples
            using this field before creating an example.
            Default: None.
        postprocessing: A Pipeline that will be applied to a list of examples
            using this field before assigning to a batch.
            Function signature: (batch(list)) -> object
            Default: None.
    """

    def __init__(self, preprocessing=None, postprocessing=None):
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing

    def preprocess(self, x):
        """ Preprocess an example if the `preprocessing` Pipeline is provided. """
        if self.preprocessing is not None:
            return self.preprocessing(x)
        else:
            return x

    def process(self, batch, *args, **kwargs):
        """ Process a list of examples to create a batch.

        Postprocess the batch with user-provided Pipeline.

        Args:
            batch (list(object)): A list of object from a batch of examples.
        Returns:
            object: Processed object given the input and custom
                postprocessing Pipeline.
        """
        if self.postprocessing is not None:
            batch = self.postprocessing(batch)
        return default_collate(batch)


class Merge(RawField):
    def __init__(self, *fields):
        super(Merge, self).__init__()
        self.fields = fields

    def preprocess(self, x):
        return tuple(f.preprocess(x) for f in self.fields)

    def process(self, batch, *args, **kwargs):
        if len(self.fields) == 1:
            batch = [batch, ]
        else:
            batch = list(zip(*batch))

        out = list(f.process(b, *args, **kwargs) for f, b in zip(self.fields, batch))
        return out


class ImageField(RawField):
    def __init__(self, preprocessing=None, postprocessing=None, loader=default_loader, transform=None):
        self.loader = loader
        self.transform = transform
        super().__init__(preprocessing, postprocessing)

    def preprocess(self, x):
        sample = self.loader(x)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


class TextField(RawField):
    def __init__(self):
        self._tokenizer = _Tokenizer()
        super(TextField, self).__init__()

    def preprocess(self, x):
        if x is None:
            return ''
        return x

    def process(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        sot_token = self._tokenizer.bos_idx
        eot_token = self._tokenizer.eos_idx
        all_tokens = [[sot_token] + self._tokenizer.encode(text) + [eot_token] for text in texts]
        result = torch.zeros(len(all_tokens), max(len(s) for s in all_tokens), dtype=torch.long)

        for i, tokens in enumerate(all_tokens):
            result[i, :len(tokens)] = torch.tensor(tokens)

        return result

    def decode(self, word_idxs):
        if isinstance(word_idxs, list) and len(word_idxs) == 0:
            return self.decode([word_idxs, ])[0]
        if isinstance(word_idxs, list) and isinstance(word_idxs[0], int):
            return self.decode([word_idxs, ])[0]
        elif isinstance(word_idxs, torch.Tensor) and word_idxs.ndimension() == 1:
            return self.decode(word_idxs.unsqueeze(0))[0]

        captions = []
        for wis in word_idxs:
            wis = wis.tolist()
            wis = list(takewhile(lambda tok: tok != self._tokenizer.eos_idx, wis))
            caption = self._tokenizer.decode(wis)
            captions.append(caption)
        return captions
    
    
class ArtEmisDetectionsField(RawField):
    def __init__(self, preprocessing=None, postprocessing=None, detections_path=None, max_detections=100):
        self.max_detections = max_detections
        
        self.detections_path = detections_path
        self.FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']
        self.features = dict()

        # load the list of ('genre/img_names', id) and create dictionary
        with open(os.path.join(detections_path,'wikiart_split.pkl'),'rb') as file:
            self.paints_ids_dict = dict(pickle.load(file))
        
        with open(os.path.join(detections_path,'tmp.csv'), "r+") as tsv_in_file:
            reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = self.FIELDNAMES)
            for item in reader:
                item['image_id'] = int(item['image_id'])
                item['image_h'] = int(item['image_h'])
                item['image_w'] = int(item['image_w'])
                item['num_boxes'] = int(item['num_boxes'])
                for field in ['boxes', 'features']:
                    data = item[field]
                    # buf = base64.decodestring(data)
                    buf = base64.b64decode(data[1:])
                    temp = np.frombuffer(buf, dtype=np.float32)
                    item[field] = temp.reshape((item['num_boxes'],-1))
                self.features[item['image_id']] = item['features']
        
        self.not_found = set()
        super(ArtEmisDetectionsField, self).__init__(preprocessing, postprocessing)

    def preprocess(self, x):

        id = self.paints_ids_dict[x]
        
        try:
            precomp_data = self.features[id]
        except KeyError:
            self.not_found.add(id)
            #warnings.warn('Could not find detections for %s (Total Missing: %d)' % (x, len(self.not_found)))
            precomp_data = np.random.rand(10, 2048)
        
        delta = self.max_detections - precomp_data.shape[0]
        if delta > 0:
            precomp_data = np.concatenate([precomp_data, np.zeros((delta, precomp_data.shape[1]))], axis=0)
        elif delta < 0:
            precomp_data = precomp_data[:self.max_detections]

        return precomp_data.astype(np.float32)
        

class EmotionField(RawField):
    def __init__(self, preprocessing=None, postprocessing=None, emotions=None):
        
        self.emotion_mapping = { key: value for value, key in enumerate(emotions)}
        super(EmotionField, self).__init__(preprocessing, postprocessing)

    def preprocess(self, x):
        return self.emotion_mapping[x]