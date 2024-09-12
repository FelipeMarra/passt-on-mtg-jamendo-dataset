import os
import numpy as np
import pickle

import torch
import torchaudio
from torch.utils import data

class AudioFolder(data.Dataset):
    def __init__(self, root, subset, tr_val='train', split=0):
        self.trval = tr_val
        self.root = root
        fn = '../../data/splits/split-%d/%s_%s_dict.pickle' % (split, subset, tr_val)
        self.get_dictionary(fn)

    def __getitem__(self, index) -> tuple[torch.Tensor, np.ndarray, str]:
        audio_path = os.path.join(self.root, self.dictionary[index]['path'])
        audio = torchaudio.load(audio_path)[0].squeeze()

        time_dim = audio.shape[0]
        time_dim_expected = 32000 * 10

        if time_dim > time_dim_expected:
            audio = audio[:time_dim_expected]
        elif time_dim < time_dim_expected:
            pad_size = time_dim_expected - time_dim
            pad = torch.zeros(pad_size, dtype=torch.float32)
            audio = torch.cat((audio, pad), dim=0)

        tags:np.ndarray = self.dictionary[index]['tags']
        return audio, tags.astype('float32'), self.dictionary[index]['path']

    def get_dictionary(self, fn):
        with open(fn, 'rb') as pf:
            dictionary = pickle.load(pf)
        self.dictionary = dictionary

    def __len__(self):
        return len(self.dictionary)

def get_audio_loader(root, subset, batch_size, tr_val='train', split=0, num_workers=0):
    data_loader = data.DataLoader(dataset=AudioFolder(root, subset, tr_val, split),
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    return data_loader

