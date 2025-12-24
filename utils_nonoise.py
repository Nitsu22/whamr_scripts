import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def read_scaled_wav(path, scaling_factor, downsample_8K=False, mono=True):
    samples, sr_orig = sf.read(path)

    if len(samples.shape) > 1 and mono:
        samples = samples[:, 0]

    if downsample_8K:
        samples = resample_poly(samples, 8000, sr_orig)
    samples *= scaling_factor
    return samples


def wavwrite_quantize(samples):
    return np.int16(np.round((2 ** 15) * samples))


def quantize(samples):
    int_samples = wavwrite_quantize(samples)
    return np.float64(int_samples) / (2 ** 15)


def wavwrite(file, samples, sr):
    """This is how the old Matlab function wavwrite() quantized to 16 bit.
    We match it here to maintain parity with the original dataset"""
    int_samples = wavwrite_quantize(samples)
    sf.write(file, int_samples, sr, subtype='PCM_16')


def append_or_truncate(s1_samples, s2_samples, min_or_max='max', start_samp_16k=0, downsample=False):
    # For no-noise version, we just need to fix the length
    # start_samp_16k is not used since there's no noise to align with
    return fix_length(s1_samples, s2_samples, min_or_max)


def append_zeros(samples, desired_length):
    samples_to_add = desired_length - len(samples)
    if len(samples.shape) == 1:
        new_zeros = np.zeros(samples_to_add)
    elif len(samples.shape) == 2:
        # Support any number of channels
        new_zeros = np.zeros((samples_to_add, samples.shape[1]))
    return np.append(samples, new_zeros, axis=0)


def fix_length(s1, s2, min_or_max='max'):
    # Fix length
    if min_or_max == 'min':
        utt_len = np.minimum(len(s1), len(s2))
        s1 = s1[:utt_len]
        s2 = s2[:utt_len]
    else:  # max
        utt_len = np.maximum(len(s1), len(s2))
        s1 = append_zeros(s1, utt_len)
        s2 = append_zeros(s2, utt_len)
    return s1, s2


def create_wham_mixes(s1_samples, s2_samples):
    mix_clean = s1_samples + s2_samples
    # mix_single and mix_both removed: they contain noise
    return mix_clean