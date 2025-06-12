# Adapted from https://github.com/NVIDIA/CleanUNet/blob/main/python_eval.py and https://github.com/ruizhecao96/CMGAN/blob/39946005d9b66fa1e824edf4bb6bc9a06088e443/src/tools/compute_metrics.py#L45C1-L73C50

# Original Copyright (c) 2022 NVIDIA CORPORATION and Copyright (c) 2022 Ruizhe Cao.
#   Licensed under the MIT license.

import math
import os
import sys
from collections import defaultdict

from scipy.linalg import toeplitz
from scipy.fftpack import fft
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy.io import wavfile
from scipy.io.wavfile import write as wavwrite

from pesq import pesq
from pystoi import stoi


def evaluate_dns(testset_path, enhanced_path, target):
    reverb = 'no'
    recenter = False
    warmup_s = 0

    noisy_files = os.listdir(enhanced_path)

    for i in tqdm(range(300)):
        try:
            rate, clean = wavfile.read(os.path.join(testset_path, "clean", "clean_fileid_{}.wav".format(i)))
            if target == 'noisy':
                rate, target_wav = wavfile.read(os.path.join(testset_path, "noisy", "noisy_fileid_{}.wav".format(i)))
            if target == 'freeform':
                for file in noisy_files:
                    if file.split('_')[-1][0:-4] == str(i):
                        rate, target_wav = wavfile.read(os.path.join(enhanced_path, file))
                        break
                if target_wav is None:
                    continue
            else:
                rate, target_wav = wavfile.read(os.path.join(enhanced_path, "enhanced_fileid_{}.wav".format(i)))
        except:
            continue


        if recenter:
            clean = clean / 2**15
            clean = clean.astype(np.float32)

            dam = target_wav.mean()
            clm = clean.mean()
            target_wav = target_wav - dam
            clean = clean - clm

            # scale with the difference in variance
            dav = target_wav.var()
            clv = clean.var()

            target_wav = target_wav * (clv / dav) + clm
            clean = clean + clm

            # Write out the recentered files
            if not os.path.isdir(os.path.join(enhanced_path, "recentered")):
                os.makedirs(os.path.join(enhanced_path, "recentered"))
                os.chmod(os.path.join(enhanced_path, "recentered"), 0o775)

            wavwrite(os.path.join(enhanced_path, "recentered", "enhanced_fileid_{}.wav".format(i)),
                     rate, target_wav)

        # Crop the first warmup seconds
        clean = clean[rate*warmup_s:]
        target_wav = target_wav[rate*warmup_s:]
    
    return eval_waveform(clean, target_wav, rate)

def eval_waveform(clean, target_wav, rate):
    result = defaultdict(int)

    length = target_wav.shape[-1]
    alpha = 0.95

    # The following metrics are calculated following https://github.com/ruizhecao96/CMGAN/blob/39946005d9b66fa1e824edf4bb6bc9a06088e443/src/tools/compute_metrics.py#L45C1-L73C50

    # compute the WSS measure
    wss_dist_vec = wss(clean, target_wav, 16000)
    wss_dist_vec = np.sort(wss_dist_vec)
    wss_dist = np.mean(wss_dist_vec[0: round(np.size(wss_dist_vec) * alpha)])

    # compute the LLR measure
    LLR_dist = llr(clean, target_wav, 16000)
    LLRs = np.sort(LLR_dist)
    LLR_len = round(np.size(LLR_dist) * alpha)
    llr_top = LLRs[0:LLR_len]
    llr_top = llr_top[~np.isnan(llr_top)] # Remove nan's
    llr_mean = np.mean(llr_top)

    # compute the SNRseg
    snr_dist, segsnr_dist = snr(clean, target_wav, 16000)
    snr_mean = snr_dist
    segSNR = np.mean(segsnr_dist)

    # compute the pesq
    pesq_mos = pesq(16000, clean, target_wav, 'wb')

    # now compute the composite measures
    CSIG = 3.093 - 1.029 * llr_mean + 0.603 * pesq_mos - 0.009 * wss_dist
    CSIG = max(1, CSIG)
    CSIG = min(5, CSIG)  # limit values to [1, 5]
    CBAK = 1.634 + 0.478 * pesq_mos - 0.007 * wss_dist + 0.063 * segSNR
    CBAK = max(1, CBAK)
    CBAK = min(5, CBAK)  # limit values to [1, 5]
    COVL = 1.594 + 0.805 * pesq_mos - 0.512 * llr_mean - 0.007 * wss_dist
    COVL = max(1, COVL)
    COVL = min(5, COVL)  # limit values to [1, 5]

    result['pesq_wb'] += pesq_mos * length  # wide band
    result['pesq_nb'] += pesq(16000, clean, target_wav, 'nb') * length  # narrow band
    result['stoi'] += stoi(clean, target_wav, rate) * length
    result['CSIG'] += CSIG * length
    result['CBAK'] += CBAK * length
    result['COVL'] += COVL * length

    result['wss_dist'] += wss_dist * length
    result['segSNR'] += segSNR * length
    result['llr_mean'] += llr_mean * length

    result['count'] += 1 * length


    return result



def wss(clean_speech, processed_speech, sample_rate):
    # Check the length of the clean and processed speech, which must be the same.
    clean_length = np.size(clean_speech)
    processed_length = np.size(processed_speech)
    if clean_length != processed_length:
        raise ValueError("Files must have same length.")

    # Global variables
    winlength = (np.round(30 * sample_rate / 1000)).astype(
        int
    )  # window length in samples
    skiprate = (np.floor(np.divide(winlength, 4))).astype(int)  # window skip in samples
    max_freq = (np.divide(sample_rate, 2)).astype(int)  # maximum bandwidth
    num_crit = 25  # number of critical bands

    USE_FFT_SPECTRUM = 1  # defaults to 10th order LP spectrum
    n_fft = (np.power(2, np.ceil(np.log2(2 * winlength)))).astype(int)
    n_fftby2 = (np.multiply(0.5, n_fft)).astype(int)  # FFT size/2
    Kmax = 20.0  # value suggested by Klatt, pg 1280
    Klocmax = 1.0  # value suggested by Klatt, pg 1280

    # Critical Band Filter Definitions (Center Frequency and Bandwidths in Hz)
    cent_freq = np.array(
        [
            50.0000,
            120.000,
            190.000,
            260.000,
            330.000,
            400.000,
            470.000,
            540.000,
            617.372,
            703.378,
            798.717,
            904.128,
            1020.38,
            1148.30,
            1288.72,
            1442.54,
            1610.70,
            1794.16,
            1993.93,
            2211.08,
            2446.71,
            2701.97,
            2978.04,
            3276.17,
            3597.63,
        ]
    )
    bandwidth = np.array(
        [
            70.0000,
            70.0000,
            70.0000,
            70.0000,
            70.0000,
            70.0000,
            70.0000,
            77.3724,
            86.0056,
            95.3398,
            105.411,
            116.256,
            127.914,
            140.423,
            153.823,
            168.154,
            183.457,
            199.776,
            217.153,
            235.631,
            255.255,
            276.072,
            298.126,
            321.465,
            346.136,
        ]
    )

    bw_min = bandwidth[0]  # minimum critical bandwidth

    # Set up the critical band filters.
    # Note here that Gaussianly shaped filters are used.
    # Also, the sum of the filter weights are equivalent for each critical band filter.
    # Filter less than -30 dB and set to zero.
    min_factor = math.exp(-30.0 / (2.0 * 2.303))  # -30 dB point of filter
    crit_filter = np.empty((num_crit, n_fftby2))
    for i in range(num_crit):
        f0 = (cent_freq[i] / max_freq) * n_fftby2
        bw = (bandwidth[i] / max_freq) * n_fftby2
        norm_factor = np.log(bw_min) - np.log(bandwidth[i])
        j = np.arange(n_fftby2)
        crit_filter[i, :] = np.exp(
            -11 * np.square(np.divide(j - np.floor(f0), bw)) + norm_factor
        )
        cond = np.greater(crit_filter[i, :], min_factor)
        crit_filter[i, :] = np.where(cond, crit_filter[i, :], 0)
    # For each frame of input speech, calculate the Weighted Spectral Slope Measure
    num_frames = int(
        clean_length / skiprate - (winlength / skiprate)
    )  # number of frames
    start = 0  # starting sample
    window = 0.5 * (
        1 - np.cos(2 * math.pi * np.arange(1, winlength + 1) / (winlength + 1))
    )

    distortion = np.empty(num_frames)
    for frame_count in range(num_frames):
        # (1) Get the Frames for the test and reference speech. Multiply by Hanning Window.
        clean_frame = clean_speech[start : start + winlength] / 32768
        processed_frame = processed_speech[start : start + winlength] / 32768
        clean_frame = np.multiply(clean_frame, window)
        processed_frame = np.multiply(processed_frame, window)
        # (2) Compute the Power Spectrum of Clean and Processed
        # if USE_FFT_SPECTRUM:
        clean_spec = np.square(np.abs(fft(clean_frame, n_fft)))
        processed_spec = np.square(np.abs(fft(processed_frame, n_fft)))

        # (3) Compute Filterbank Output Energies (in dB scale)
        clean_energy = np.matmul(crit_filter, clean_spec[0:n_fftby2])
        processed_energy = np.matmul(crit_filter, processed_spec[0:n_fftby2])

        clean_energy = 10 * np.log10(np.maximum(clean_energy, 1e-10))
        processed_energy = 10 * np.log10(np.maximum(processed_energy, 1e-10))

        # (4) Compute Spectral Slope (dB[i+1]-dB[i])
        clean_slope = clean_energy[1:num_crit] - clean_energy[0 : num_crit - 1]
        processed_slope = (
            processed_energy[1:num_crit] - processed_energy[0 : num_crit - 1]
        )

        # (5) Find the nearest peak locations in the spectra to each critical band.
        #     If the slope is negative, we search to the left. If positive, we search to the right.
        clean_loc_peak = np.empty(num_crit - 1)
        processed_loc_peak = np.empty(num_crit - 1)

        for i in range(num_crit - 1):
            # find the peaks in the clean speech signal
            if clean_slope[i] > 0:  # search to the right
                n = i
                while (n < num_crit - 1) and (clean_slope[n] > 0):
                    n = n + 1
                clean_loc_peak[i] = clean_energy[n - 1]
            else:  # search to the left
                n = i
                while (n >= 0) and (clean_slope[n] <= 0):
                    n = n - 1
                clean_loc_peak[i] = clean_energy[n + 1]

            # find the peaks in the processed speech signal
            if processed_slope[i] > 0:  # search to the right
                n = i
                while (n < num_crit - 1) and (processed_slope[n] > 0):
                    n = n + 1
                processed_loc_peak[i] = processed_energy[n - 1]
            else:  # search to the left
                n = i
                while (n >= 0) and (processed_slope[n] <= 0):
                    n = n - 1
                processed_loc_peak[i] = processed_energy[n + 1]

        # (6) Compute the WSS Measure for this frame. This includes determination of the weighting function.
        dBMax_clean = np.max(clean_energy)
        dBMax_processed = np.max(processed_energy)
        """
        The weights are calculated by averaging individual weighting factors from the clean and processed frame.
        These weights W_clean and W_processed should range from 0 to 1 and place more emphasis on spectral peaks
        and less emphasis on slope differences in spectral valleys.
        This procedure is described on page 1280 of Klatt's 1982 ICASSP paper.
        """
        Wmax_clean = np.divide(
            Kmax, Kmax + dBMax_clean - clean_energy[0 : num_crit - 1]
        )
        Wlocmax_clean = np.divide(
            Klocmax, Klocmax + clean_loc_peak - clean_energy[0 : num_crit - 1]
        )
        W_clean = np.multiply(Wmax_clean, Wlocmax_clean)

        Wmax_processed = np.divide(
            Kmax, Kmax + dBMax_processed - processed_energy[0 : num_crit - 1]
        )
        Wlocmax_processed = np.divide(
            Klocmax, Klocmax + processed_loc_peak - processed_energy[0 : num_crit - 1]
        )
        W_processed = np.multiply(Wmax_processed, Wlocmax_processed)

        W = np.divide(np.add(W_clean, W_processed), 2.0)
        slope_diff = np.subtract(clean_slope, processed_slope)[0 : num_crit - 1]
        distortion[frame_count] = np.dot(W, np.square(slope_diff)) / np.sum(W)
        # this normalization is not part of Klatt's paper, but helps to normalize the measure.
        # Here we scale the measure by the sum of the weights.
        start = start + skiprate
    return distortion


def llr(clean_speech, processed_speech, sample_rate):
    # Check the length of the clean and processed speech.  Must be the same.
    clean_length = np.size(clean_speech)
    processed_length = np.size(processed_speech)
    if clean_length != processed_length:
        raise ValueError("Both Speech Files must be same length.")

    # Global Variables
    winlength = (np.round(30 * sample_rate / 1000)).astype(
        int
    )  # window length in samples
    skiprate = (np.floor(winlength / 4)).astype(int)  # window skip in samples
    if sample_rate < 10000:
        P = 10  # LPC Analysis Order
    else:
        P = 16  # this could vary depending on sampling frequency.

    # For each frame of input speech, calculate the Log Likelihood Ratio
    num_frames = int((clean_length - winlength) / skiprate)  # number of frames
    start = 0  # starting sample
    window = 0.5 * (
        1 - np.cos(2 * math.pi * np.arange(1, winlength + 1) / (winlength + 1))
    )

    distortion = np.empty(num_frames)
    for frame_count in range(num_frames):
        # (1) Get the Frames for the test and reference speech. Multiply by Hanning Window.
        clean_frame = clean_speech[start : start + winlength]
        processed_frame = processed_speech[start : start + winlength]
        clean_frame = np.multiply(clean_frame, window)
        processed_frame = np.multiply(processed_frame, window)

        # (2) Get the autocorrelation lags and LPC parameters used to compute the LLR measure.
        R_clean, Ref_clean, A_clean = lpcoeff(clean_frame, P)
        R_processed, Ref_processed, A_processed = lpcoeff(processed_frame, P)

        # (3) Compute the LLR measure
        numerator = np.dot(np.matmul(A_processed, toeplitz(R_clean)), A_processed)
        denominator = np.dot(np.matmul(A_clean, toeplitz(R_clean)), A_clean)
        distortion[frame_count] = np.log(numerator / denominator)
        start = start + skiprate
    return distortion


def lpcoeff(speech_frame, model_order):
    # (1) Compute Autocorrelation Lags
    winlength = np.size(speech_frame)
    R = np.empty(model_order + 1)
    E = np.empty(model_order + 1)
    for k in range(model_order + 1):
        R[k] = np.dot(speech_frame[0 : winlength - k], speech_frame[k:winlength])

    # (2) Levinson-Durbin
    a = np.ones(model_order)
    a_past = np.empty(model_order)
    rcoeff = np.empty(model_order)
    E[0] = R[0]
    for i in range(model_order):
        a_past[0:i] = a[0:i]
        sum_term = np.dot(a_past[0:i], R[i:0:-1])
        rcoeff[i] = (R[i + 1] - sum_term) / E[i]
        a[i] = rcoeff[i]
        if i == 0:
            a[0:i] = a_past[0:i] - np.multiply(a_past[i - 1 : -1 : -1], rcoeff[i])
        else:
            a[0:i] = a_past[0:i] - np.multiply(a_past[i - 1 :: -1], rcoeff[i])
        E[i + 1] = (1 - rcoeff[i] * rcoeff[i]) * E[i]
    acorr = R
    refcoeff = rcoeff
    lpparams = np.concatenate((np.array([1]), -a))
    return acorr, refcoeff, lpparams


def snr(clean_speech, processed_speech, sample_rate):
    # Check the length of the clean and processed speech. Must be the same.
    clean_length = len(clean_speech)
    processed_length = len(processed_speech)
    if clean_length != processed_length:
        raise ValueError("Both Speech Files must be same length.")

    overall_snr = 10 * np.log10(
        np.sum(np.square(clean_speech))
        / np.sum(np.square(clean_speech - processed_speech))
    )

    # Global Variables
    winlength = round(30 * sample_rate / 1000)  # window length in samples
    skiprate = math.floor(winlength / 4)  # window skip in samples
    MIN_SNR = -10  # minimum SNR in dB
    MAX_SNR = 35  # maximum SNR in dB

    # For each frame of input speech, calculate the Segmental SNR
    num_frames = int(
        clean_length / skiprate - (winlength / skiprate)
    )  # number of frames
    start = 0  # starting sample
    window = 0.5 * (
        1 - np.cos(2 * math.pi * np.arange(1, winlength + 1) / (winlength + 1))
    )

    segmental_snr = np.empty(num_frames)
    EPS = np.spacing(1)
    for frame_count in range(num_frames):
        # (1) Get the Frames for the test and reference speech. Multiply by Hanning Window.
        clean_frame = clean_speech[start : start + winlength]
        processed_frame = processed_speech[start : start + winlength]
        clean_frame = np.multiply(clean_frame, window)
        processed_frame = np.multiply(processed_frame, window)

        # (2) Compute the Segmental SNR
        signal_energy = np.sum(np.square(clean_frame))
        noise_energy = np.sum(np.square(clean_frame - processed_frame))
        segmental_snr[frame_count] = 10 * math.log10(
            signal_energy / (noise_energy + EPS) + EPS
        )
        segmental_snr[frame_count] = max(segmental_snr[frame_count], MIN_SNR)
        segmental_snr[frame_count] = min(segmental_snr[frame_count], MAX_SNR)

        start = start + skiprate

    return overall_snr, segmental_snr




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='dns', help='dataset')
    parser.add_argument('-e', '--enhanced_path', type=str, help='enhanced audio path')
    parser.add_argument('-t', '--testset_path', type=str, help='testset path',
                        default='/media/sjoerd/storage/boeken/jaar7/thesis/DNS-Challenge/datasets/test_set/synthetic/no_reverb')
    args = parser.parse_args()

    enhanced_path = args.enhanced_path
    testset_path = args.testset_path
    target = 'freeform'

    if args.dataset == 'dns':
        result = evaluate_dns(testset_path, enhanced_path, target)
        
    # logging
    for key in result:
        if key != 'count':
            print('{} = {:.3f}'.format(key, result[key]/result['count']), end=", ")
