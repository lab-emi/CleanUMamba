# Copyright (c) 2022 NVIDIA CORPORATION. 
#   Licensed under the MIT license.
import logging
import os
import numpy as np

from scipy.io.wavfile import read as wavread
import warnings

warnings.filterwarnings("ignore")

import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

# Audio quantization functions form S4
# from src.Network.s4.src.dataloaders.audio import linear_encode, linear_decode, mu_law_encode, mu_law_decode, \
#     minmax_scale

import random

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

from torchvision import datasets, models, transforms
import torchaudio


class CleanNoisyPairDataset(Dataset):
    """
    Create a Dataset of clean and noisy audio pairs. 
    Each element is a tuple of the form (clean waveform, noisy waveform, (min, max value of the noisy waveform), file_id)
    """

    def __init__(self, root='./', subset='training', crop_length_sec=0, dataset="dns", quantization=None, bits=16):
        super(CleanNoisyPairDataset).__init__()

        assert subset is None or subset in ["training", "testing"]
        self.crop_length_sec = crop_length_sec
        self.subset = subset

        N_clean = len(os.listdir(os.path.join(root, 'training_set/clean')))
        N_noisy = len(os.listdir(os.path.join(root, 'training_set/noisy')))
        assert N_clean == N_noisy
        assert N_clean > 0, f"Number of dataset files: {N_clean}"

        self.bits = bits
        self.create_quantizer(quantization)

        if dataset == "VCTK-DEMAND":
            names = os.listdir(os.path.join(root, 'training_set/clean'))
            self.files = [(os.path.join(root, 'training_set/clean', name),
                           os.path.join(root, 'training_set/noisy', name)) for name in names]
        elif subset == "training":
            self.files = [(os.path.join(root, 'training_set/clean', 'fileid_{}.wav'.format(i)),
                           os.path.join(root, 'training_set/noisy', 'fileid_{}.wav'.format(i))) for i in range(N_clean)]
        elif subset == "testing":
            sortkey = lambda name: '_'.join(name.split('_')[-2:])  # specific for dns due to test sample names
            _p = os.path.join(root, 'datasets/test_set/synthetic/no_reverb')  # path for DNS

            clean_files = os.listdir(os.path.join(_p, 'clean'))
            noisy_files = os.listdir(os.path.join(_p, 'noisy'))

            clean_files.sort(key=sortkey)
            noisy_files.sort(key=sortkey)

            self.files = []
            for _c, _n in zip(clean_files, noisy_files):
                assert sortkey(_c) == sortkey(_n)
                self.files.append((os.path.join(_p, 'clean', _c),
                                   os.path.join(_p, 'noisy', _n)))
            self.crop_length_sec = 0

        else:
            raise NotImplementedError

    def enable_quantization(self, quantization: str, bits):
        self.create_quantizer('linear')
        self.bits = bits

    def disable_quantization(self):
        self.create_quantizer(None)
        self.bits = 16

    def dequantize(self, x):
        return self.dequantizer(x, self.bits)

    def create_quantizer(self, quantization: str):
        if quantization == 'linear':
            # self.quantizer = linear_encode
            # self.dequantizer = linear_decode
            # self.qtype = torch.long
            raise NotImplementedError
        elif quantization == 'mu-law':
            # self.quantizer = mu_law_encode
            # self.dequantizer = mu_law_decode
            # self.qtype = torch.long
            raise NotImplementedError
        elif quantization is None:
            self.quantizer = lambda x, bits: x
            self.dequantizer = lambda x, bits: x
            self.qtype = torch.float
        else:
            print(f"Invalid quantization type: {quantization}")
            raise ValueError('Invalid quantization type')

    def __getitem__(self, n):
        fileid = self.files[n]
        clean_audio, sample_rate = torchaudio.load(fileid[0])
        noisy_audio, sample_rate = torchaudio.load(fileid[1])
        clean_audio, noisy_audio = clean_audio.squeeze(0), noisy_audio.squeeze(0)
        assert len(clean_audio) == len(noisy_audio), f"Length mismatch in {fileid}, {len(clean_audio)} vs {len(noisy_audio)}"

        crop_length = int(self.crop_length_sec * sample_rate)
        # assert crop_length < len(clean_audio), f"{crop_length} < {len(clean_audio)}"

        # random crop
        length = len(clean_audio)
        if crop_length > length:
            units = crop_length // length
            clean_ds_final = []
            noisy_ds_final = []
            for i in range(units):
                clean_ds_final.append(clean_audio)
                noisy_ds_final.append(noisy_audio)
            clean_ds_final.append(clean_audio[: crop_length % length])
            noisy_ds_final.append(noisy_audio[: crop_length % length])
            clean_audio = torch.cat(clean_ds_final, dim=-1)
            noisy_audio = torch.cat(noisy_ds_final, dim=-1)
        elif self.subset != 'testing' and crop_length > 0:
            start = np.random.randint(low=0, high=len(clean_audio) - crop_length + 1)
            clean_audio = clean_audio[start:(start + crop_length)]
            noisy_audio = noisy_audio[start:(start + crop_length)]

        # Unsqueeze to (1, L, 1)
        clean_audio, noisy_audio = clean_audio.unsqueeze(0).unsqueeze(-1), noisy_audio.unsqueeze(0).unsqueeze(-1)

        # Quantized signal
        min_val = -1
        max_val = 1
        if self.qtype is torch.long:
            min_val = torch.amin(noisy_audio, dim=(1, 2), keepdim=True)
            max_val = torch.amax(noisy_audio, dim=(1, 2), keepdim=True)
        noisy_audio = self.quantizer(noisy_audio, self.bits)

        # Squeeze to (1, L)
        clean_audio, noisy_audio = clean_audio.squeeze(-1), noisy_audio.squeeze(-1)

        return clean_audio, noisy_audio, (min_val, max_val), fileid

    def __len__(self):
        return len(self.files)


def load_CleanNoisyPairDataset(root="/media/sjoerd/storage/boeken/jaar7/thesis/DNS-Challenge",
                               subset="training",
                               dataset="dns",
                               crop_length_sec=10,
                               batch_size=1,
                               quantization=None,
                               bits=32,
                               sample_rate=16000,
                               shuffle=True,
                               num_gpus=1,
                               num_workers=0):
    """
    Get dataloader with distributed sampling
    """
    if root == "/media/sjoerd/storage/boeken/jaar7/thesis/DNS-Challenge":
        logging.warning("[util/dataset.py:load_CleanNoisyPairDataset()] replace the root with your own DNS-Challenge dataset path")

    dataset = CleanNoisyPairDataset(root=root, subset=subset, crop_length_sec=crop_length_sec,
                                    dataset=dataset, quantization=quantization, bits=bits)
    kwargs = {"batch_size": batch_size, "num_workers": num_workers, "pin_memory": False, "drop_last": False}
    # put num workers on 0 for debug ability and 4 for speed

    if num_gpus > 1:
        train_sampler = DistributedSampler(dataset)
        dataloader = torch.utils.data.DataLoader(dataset, sampler=train_sampler, **kwargs)
    else:
        dataloader = torch.utils.data.DataLoader(dataset, sampler=None, shuffle=shuffle, **kwargs)

    return dataloader


class NosyOnlyDataset(Dataset):
    """
    Create a Dataset of noisy audio pairs.
    Each element is a tuple of the form (None, noisy waveform, file_id)
    """

    def __init__(self, folder='./'):
        super(CleanNoisyPairDataset).__init__()

        self.folder = folder

        self.noisy_files = os.listdir(folder)
        self.crop_length_sec = 0

    def __getitem__(self, n):
        fileid = self.noisy_files[n]
        noisy_audio, sample_rate = torchaudio.load(os.path.join(self.folder, fileid))
        noisy_audio = noisy_audio[0, :]
        return noisy_audio, noisy_audio, (-1, 1), fileid

    def __len__(self):
        return len(self.noisy_files)


def load_NoisyDataset(folder, batch_size, sample_rate, num_gpus=1):
    """
    Get dataloader with distributed sampling
    """
    dataset = NosyOnlyDataset(folder=folder)
    kwargs = {"batch_size": batch_size, "num_workers": 0, "pin_memory": False, "drop_last": False}

    if num_gpus > 1:
        train_sampler = DistributedSampler(dataset)
        dataloader = torch.utils.data.DataLoader(dataset, sampler=train_sampler, **kwargs)
    else:
        dataloader = torch.utils.data.DataLoader(dataset, sampler=None, shuffle=True, **kwargs)

    return dataloader


if __name__ == '__main__':
    import json

    with open('./configs/config.json') as f:
        data = f.read()
    config = json.loads(data)
    trainset_config = config["trainset_config"]

    trainloader = load_CleanNoisyPairDataset(**trainset_config, subset='training', batch_size=2, num_gpus=1)
    testloader = load_CleanNoisyPairDataset(**trainset_config, subset='testing', batch_size=2, num_gpus=1)
    print(len(trainloader), len(testloader))

    for clean_audio, noisy_audio, fileid in trainloader:
        clean_audio = clean_audio.cuda()
        noisy_audio = noisy_audio.cuda()
        print(clean_audio.shape, noisy_audio.shape, fileid)
        break
