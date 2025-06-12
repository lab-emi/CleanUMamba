import logging
import os
import argparse
import json

from tqdm import tqdm
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

import random

from src.util.python_eval import wss, llr, snr, eval_waveform

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)


def validate(net, testset_path, quantize_audio_input=False, network_processor=None, validation=True, VCTK_DEMAND=False, metrics=("pesq_wb", "pesq_nb", "stoi", "DNSMOS"), test_factor=1.0, crop=None):
    from scipy.io import wavfile

    from src.util.util import sampling
    import torchaudio

    if validation:
        noisy_files = os.listdir(os.path.join(testset_path, "noisy"))

    result = {
        'Test/pesq_wb': 0,
        'Test/pesq_nb': 0,
        'Test/stoi': 0,
        'Test/CSIG': 0,
        'Test/CBAK': 0,
        'Test/COVL': 0,
        'Test/wss_dist': 0,
        'Test/segSNR': 0,
        'Test/llr_mean': 0,
        'Test/count': 0
    }

    n = len(noisy_files)
    if not VCTK_DEMAND:
        n *= 2

    for i in (pbar := tqdm(range(int(n * test_factor)))):
        try:
            if VCTK_DEMAND:
                rate, clean = wavfile.read(os.path.join(testset_path, "clean" ,noisy_files[i]))
            else:
                rate, clean = wavfile.read(os.path.join(testset_path, "clean", "clean_fileid_{}.wav".format(
                    i) if validation else "fileid_{}.wav".format(i)))

            # rate, noisy_audio = wavfile.read(os.path.join(testset_path, "noisy", "noisy_fileid_{}.wav".format(i)))
            if VCTK_DEMAND:
                noisy_audio, rate = torchaudio.load(os.path.join(testset_path, "noisy", noisy_files[i]))
            elif validation:
                for file in noisy_files:
                    if file.split('_')[-1][0:-4] == str(i):
                        noisy_audio, rate = torchaudio.load(os.path.join(testset_path, "noisy", file))
                        break
            else:
                noisy_audio, rate = torchaudio.load(os.path.join(testset_path, "noisy", "fileid_{}.wav".format(i)))

            if noisy_audio is None:
                continue

            device = 'cuda'
            half = False
            if getattr(net, 'parameters', None) is not None:
                for p in net.parameters():
                    device = p.device
                    half = p.dtype is torch.float16
                    break

            if half:
                noisy_audio = noisy_audio.half()

            try:
                noisy_audio_input = network_processor.prepare_input(noisy_audio)
            except:
                # print("no prepare input")
                noisy_audio_input = noisy_audio

            denoised_audio = sampling(net, torch.tensor(noisy_audio_input).to(device), split_sampling=False)

            try:
                denoised_audio = network_processor.finish_output(denoised_audio)
            except:
                pass

            if half:
                denoised_audio = denoised_audio.float()

            # wavwrite(os.path.join(testset_path, "noisy", "", "noisy_fileid_{}.wav".format(i))),
            #          rate, denoised_audio.squeeze().cpu().numpy())
            denoised_audio *= 32767 # Scale from float32 (-1,1) to int16 (-32767, 32767)
            denoised_audio = denoised_audio.squeeze().to(torch.int16).cpu().numpy()

        except Exception as error:
            if type(error) is not FileNotFoundError:
                print(error)
            continue

        if crop is not None:
            clean = clean[:crop * rate]
            denoised_audio = denoised_audio[:crop * rate]

        eval_result = eval_waveform(clean, denoised_audio, rate)
        for key in eval_result.keys():
            result['Test/' + key] += eval_result[key]

        pbar.set_postfix({'stoi': result['Test/stoi']/result['Test/count']})

    return result


def test_validation():
    from src.examples.loading_pretrained_models import load_pretrained_CleanUMamba
    model = load_pretrained_CleanUMamba("../../checkpoints/pruned/CleanUMamba-3N-E8_pruned-5M.pkl")

    testset_path="/media/sjoerd/storage/boeken/jaar7/thesis/DNS-Challenge/datasets/test_set/synthetic/no_reverb"
    if testset_path=="/media/sjoerd/storage/boeken/jaar7/thesis/DNS-Challenge/datasets/test_set/synthetic/no_reverb":
        logging.warning("[util/denoise_eval.py:test_validation()] replace the root with your own DNS-Challenge dataset path")

    print("DNS")
    results = validate(model, testset_path=testset_path,
                       test_factor=1)
    for key in results.keys():
        if key != 'Test/count':
            print(f"Model {key[5:]}={results[key] / results['Test/count']:.4f}")




if __name__ == "__main__":
    test_validation()
