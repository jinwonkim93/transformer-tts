import librosa
import librosa.filters
import librosa.util as librosa_util
import numpy as np
import soundfile
import torch
from scipy import signal
from scipy.io import wavfile
from scipy.signal import get_window

# Conversions
_mel_basis = None
_inv_mel_basis = None


def window_sumsquare(
    window,
    n_frames,
    hop_length=200,
    win_length=800,
    n_fft=800,
    dtype=np.float32,
    norm=None,
):
    """
    # from librosa 0.6
    Compute the sum-square envelope of a window function at a given hop length.

    This is used to estimate modulation effects induced by windowing
    observations in short-time fourier transforms.

    Parameters
    ----------
    window : string, tuple, number, callable, or list-like
        Window specification, as in `get_window`

    n_frames : int > 0
        The number of analysis frames

    hop_length : int > 0
        The number of samples to advance between frames

    win_length : [optional]
        The length of the window function.  By default, this matches `n_fft`.

    n_fft : int > 0
        The length of each analysis frame.

    dtype : np.dtype
        The data type of the output

    Returns
    -------
    wss : np.ndarray, shape=`(n_fft + hop_length * (n_frames - 1))`
        The sum-squared envelope of the window function
    """
    if win_length is None:
        win_length = n_fft

    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)

    # Compute the squared window at the desired length
    win_sq = get_window(window, win_length, fftbins=True)
    win_sq = librosa_util.normalize(win_sq, norm=norm) ** 2
    win_sq = librosa_util.pad_center(win_sq, n_fft)

    # Fill the envelope
    for i in range(n_frames):
        sample = i * hop_length
        x[sample : min(n, sample + n_fft)] += win_sq[: max(0, min(n_fft, n - sample))]
    return x


def griffin_lim(magnitudes, stft_fn, n_iters=30):
    """
    PARAMS
    ------
    magnitudes: spectrogram magnitudes
    stft_fn: STFT class with transform (STFT) and inverse (ISTFT) methods
    """

    angles = np.angle(np.exp(2j * np.pi * np.random.rand(*magnitudes.size())))
    angles = angles.astype(np.float32)
    angles = torch.autograd.Variable(torch.from_numpy(angles))
    signal = stft_fn.inverse(magnitudes, angles).squeeze(1)

    for i in range(n_iters):
        _, angles = stft_fn.transform(signal)
        signal = stft_fn.inverse(magnitudes, angles).squeeze(1)
    return signal


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C


def load_wav(path, sr):
    return librosa.core.load(path, sr=sr)[0]


def save_audio_to_wav(audio, file_name="test.wav", sampling_rate=32000):

    soundfile.write(
        file_name, audio, sampling_rate, format="WAV", endian="LITTLE", subtype="PCM_16"
    )


def save_wav(wav, path, sr):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    # proposed by @dsmiller
    wavfile.write(path, sr, wav.astype(np.int16))


def save_wavenet_wav(wav, path, sr):
    librosa.output.write_wav(path, wav, sr=sr)


def preemphasis(wav, k, preemphasize=True):
    if preemphasize:
        return signal.lfilter([1, -k], [1], wav)
    return wav


def inv_preemphasis(wav, k, inv_preemphasize=True):
    if inv_preemphasize:
        return signal.lfilter([1], [1, -k], wav)
    return wav


# From https://github.com/r9y9/wavenet_vocoder/blob/master/audio.py
def start_and_end_indices(quantized, silence_threshold=2):
    for start in range(quantized.size):
        if abs(quantized[start] - 127) > silence_threshold:
            break
    for end in range(quantized.size - 1, 1, -1):
        if abs(quantized[end] - 127) > silence_threshold:
            break

    assert abs(quantized[start] - 127) > silence_threshold
    assert abs(quantized[end] - 127) > silence_threshold

    return start, end


def trim_silence(wav, trim_top_db, trim_fft_size, trim_hop_size):
    """Trim leading and trailing silence
    Useful for M-AILABS dataset if we choose to trim the extra 0.5 silence at beginning and end.
    """
    # Thanks @begeekmyfriend and @lautjy for pointing out the params contradiction. These params are separate and tunable per dataset.
    return librosa.effects.trim(
        wav, top_db=trim_top_db, frame_length=trim_fft_size, hop_length=trim_hop_size
    )[0]


def linearspectrogram(
    wav,
    preemphasize,
    preemphasis_k,
    ref_level_db,
    max_abs_value,
    min_level_db,
    signal_normalization,
    allow_clipping_in_normalization,
    symmetric_mels,
):
    D = _stft(preemphasis(wav, preemphasis_k, preemphasize))
    S = _amp_to_db(np.abs(D), min_level_db=min_level_db) - ref_level_db
    if signal_normalization:
        return _normalize(
            S,
            max_abs_value=max_abs_value,
            min_level_db=min_level_db,
            allow_clipping_in_normalization=allow_clipping_in_normalization,
            symmetric_mels=symmetric_mels,
        )
    return S


def melspectrogram(
    wav,
    preemphasize,
    preemphasis_k,
    ref_level_db,
    max_abs_value,
    min_level_db,
    signal_normalization,
    allow_clipping_in_normalization,
    symmetric_mels,
):
    D = _stft(preemphasis(wav, preemphasis_k, preemphasize))
    S = _amp_to_db(_linear_to_mel(np.abs(D), min_level_db=min_level_db)) - ref_level_db
    if signal_normalization:
        return _normalize(
            S,
            max_abs_value=max_abs_value,
            min_level_db=min_level_db,
            allow_clipping_in_normalization=allow_clipping_in_normalization,
            symmetric_mels=symmetric_mels,
        )
    return S


def inv_linear_spectrogram(
    linear_spectrogram,
    preemphasize,
    preemphasis_k,
    ref_level_db,
    max_abs_value,
    min_level_db,
    allow_clipping_in_normalization,
    symmetric_mels,
    power,
    signal_normalization,
    use_lws,
):
    """Converts linear spectrogram to waveform using librosa"""
    if signal_normalization:
        D = _denormalize(
            linear_spectrogram,
            max_abs_value=max_abs_value,
            min_level_db=min_level_db,
            allow_clipping_in_normalization=allow_clipping_in_normalization,
            symmetric_mels=symmetric_mels,
        )
    else:
        D = linear_spectrogram

    S = _db_to_amp(D + ref_level_db)  # Convert back to linear

    if use_lws:
        processor = _lws_processor()
        D = processor.run_lws(S.astype(np.float64).T ** power)
        y = processor.istft(D).astype(np.float32)
        return inv_preemphasis(y, preemphasis_k, preemphasize)
    else:
        return inv_preemphasis(_griffin_lim(S**power), preemphasis_k, preemphasize)


def inv_mel_spectrogram(
    mel_spectrogram,
    preemphasize,
    preemphasis_k,
    ref_level_db,
    signal_normalization,
    power,
    use_lws,
):
    """Converts mel spectrogram to waveform using librosa"""
    if signal_normalization:
        D = _denormalize(mel_spectrogram)
    else:
        D = mel_spectrogram

    S = _mel_to_linear(_db_to_amp(D + ref_level_db), hparams)  # Convert back to linear

    if use_lws:
        processor = _lws_processor()
        D = processor.run_lws(S.astype(np.float64).T ** power)
        y = processor.istft(D).astype(np.float32)
        return inv_preemphasis(y, preemphasis_k, preemphasize)
    else:
        return inv_preemphasis(_griffin_lim(S**power), preemphasis_k, preemphasize)


def _lws_processor(n_fft, win_size, hop_size):
    import lws

    return lws.lws(n_fft, hop_size, fftsize=win_size, mode="speech")


def _griffin_lim(S, griffin_lim_iters, n_fft, win_size, hop_size, use_lws):
    """librosa implementation of Griffin-Lim
    Based on https://github.com/librosa/librosa/issues/434
    """
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)
    y = _istft(S_complex * angles)
    for i in range(griffin_lim_iters):
        angles = np.exp(1j * np.angle(_stft(y, n_fft, win_size, hop_size, use_lws)))
        y = _istft(S_complex * angles)
    return y


def _stft(y, n_fft, win_size, hop_size, use_lws):

    if use_lws:
        return _lws_processor().stft(y).T
    else:
        return librosa.stft(
            y=y,
            n_fft=n_fft,
            hop_length=hop_size,
            win_length=win_size,
            pad_mode="constant",
        )


def _istft(y, win_size, hop_size):
    return librosa.istft(y, hop_length=hop_size, win_length=win_size)


##########################################################
# Those are only correct when using lws!!! (This was messing with Wavenet quality for a long time!)
def num_frames(length, fsize, fshift):
    """Compute number of time frames of spectrogram"""
    pad = fsize - fshift
    if length % fshift == 0:
        M = (length + pad * 2 - fsize) // fshift + 1
    else:
        M = (length + pad * 2 - fsize) // fshift + 2
    return M


def pad_lr(x, fsize, fshift):
    """Compute left and right padding"""
    M = num_frames(len(x), fsize, fshift)
    pad = fsize - fshift
    T = len(x) + 2 * pad
    r = (M - 1) * fshift + fsize - T
    return pad, pad + r


##########################################################
# Librosa correct padding
def librosa_pad_lr(x, fsize, fshift):
    return 0, (x.shape[0] // fshift + 1) * fshift - x.shape[0]


def _linear_to_mel(spectogram, sampling_rate, n_fft, num_mels, fmax, fmin):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis(sampling_rate, n_fft, num_mels, fmax, fmin)
    return np.dot(_mel_basis, spectogram)


def _mel_to_linear(mel_spectrogram, sampling_rate, n_fft, num_mels, fmax, fmin):
    global _inv_mel_basis
    if _inv_mel_basis is None:
        _inv_mel_basis = np.linalg.pinv(
            _build_mel_basis(sampling_rate, n_fft, num_mels, fmax, fmin)
        )
    return np.maximum(1e-10, np.dot(_inv_mel_basis, mel_spectrogram))


def _build_mel_basis(sampling_rate, n_fft, num_mels, fmax, fmin):
    assert fmax <= sampling_rate // 2
    return librosa.filters.mel(
        sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
    )
    assert fmax <= sampling_rate // 2
    return librosa.filters.mel(
        sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
    )


def _amp_to_db(x, min_level_db=-100):
    min_level = np.exp(min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))


def _db_to_amp(x):
    return np.power(10.0, (x) * 0.05)


def _normalize(
    S,
    max_abs_value=4.0,
    min_level_db=-100,
    allow_clipping_in_normalization=True,
    symmetric_mels=True,
):
    if allow_clipping_in_normalization:
        if symmetric_mels:
            return np.clip(
                (2 * max_abs_value) * ((S - min_level_db) / (-min_level_db))
                - max_abs_value,
                -max_abs_value,
                max_abs_value,
            )
        else:
            return np.clip(
                max_abs_value * ((S - min_level_db) / (-min_level_db)), 0, max_abs_value
            )

    assert S.max() <= 0 and S.min() - min_level_db >= 0
    if symmetric_mels:
        return (2 * max_abs_value) * (
            (S - min_level_db) / (-min_level_db)
        ) - max_abs_value
    else:
        return max_abs_value * ((S - min_level_db) / (-min_level_db))


def _denormalize(
    D,
    max_abs_value=4.0,
    min_level_db=-100,
    allow_clipping_in_normalization=True,
    symmetric_mels=True,
):
    if allow_clipping_in_normalization:
        if symmetric_mels:
            return (
                (np.clip(D, -max_abs_value, max_abs_value) + max_abs_value)
                * -min_level_db
                / (2 * max_abs_value)
            ) + min_level_db
        else:
            return (
                np.clip(D, 0, max_abs_value) * -min_level_db / max_abs_value
            ) + min_level_db

    if symmetric_mels:
        return (
            (D + max_abs_value) * -min_level_db / (2 * max_abs_value)
        ) + min_level_db
    else:
        return (D * -min_level_db / max_abs_value) + min_level_db


def _custom_griffin_lim(
    spec, sampling_rate, hop_length, win_length, n_fft, num_mels, fmax, fmin
):
    spec = librosa.db_to_amplitude(spec)  # [linear or mel dimension, time sequence]

    if spec.shape[0] == ((sampling_rate // 20) // 2) + 1:
        spec = spec
    else:
        inv_mel_filter = np.linalg.pinv(
            _build_mel_basis(sampling_rate, n_fft, num_mels, fmax, fmin)
        )
        if inv_mel_filter.shape[-1] == spec.shape[-1]:
            spec = spec.T
        spec = np.maximum(1e-10, np.dot(inv_mel_filter, spec))

    return librosa.core.griffinlim(spec, hop_length=hop_length, win_length=win_length)
