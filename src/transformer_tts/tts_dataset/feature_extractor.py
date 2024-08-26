import numpy as np
from .audio_utils import preemphasis, _stft, _amp_to_db, _normalize, _linear_to_mel


class AudioFeatureExtractor:

    def __init__(
        self,
        max_wav_value=32768.0,
        sampling_rate=22050,
        filter_length=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=8000.0,
        preemphasize=True,
        preemphasis=0.97,
        ref_level_db=20,
        min_level_db=-100,
        signal_normalization=True,
        allow_clipping_in_normalization=True,
        symmetric_mels=True,
        use_lws=False,
        frame_shift_ms=None,
        max_abs_value=4.0,
    ):
        self.max_wav_value = max_wav_value
        self.sampling_rate = sampling_rate
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mel_channels = n_mel_channels
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.preemphasize = preemphasize
        self.preemphasis = preemphasis
        self.ref_level_db = ref_level_db
        self.min_level_db = min_level_db
        self.signal_normalization = signal_normalization
        self.allow_clipping_in_normalization = allow_clipping_in_normalization
        self.symmetric_mels = symmetric_mels
        self.use_lws = use_lws
        self.frame_shift_ms = frame_shift_ms
        self.max_abs_value = max_abs_value
        self.hop_size = hop_length  # Default to hop_length, overridden if frame_shift_ms is provided

    def get_hop_size(self):
        if self.hop_size is None:
            assert self.frame_shift_ms is not None
            self.hop_size = int(self.frame_shift_ms / 1000 * self.sampling_rate)
        return self.hop_size

    def __call__(self, audio: np.array):
        raise NotImplementedError


class MelAudioFeatureExtractor(AudioFeatureExtractor):
    def __init__(
        self,
        max_wav_value=32768.0,
        sampling_rate=22050,
        filter_length=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=8000.0,
        preemphasize=True,
        preemphasis=0.97,
        ref_level_db=20,
        min_level_db=-100,
        signal_normalization=True,
        allow_clipping_in_normalization=True,
        symmetric_mels=True,
        use_lws=False,
        frame_shift_ms=None,
        max_abs_value=4.0,
    ):
        super().__init__(  # Call the superclass constructor
            max_wav_value=max_wav_value,
            sampling_rate=sampling_rate,
            filter_length=filter_length,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            mel_fmin=mel_fmin,
            mel_fmax=mel_fmax,
            preemphasize=preemphasize,
            preemphasis=preemphasis,
            ref_level_db=ref_level_db,
            min_level_db=min_level_db,
            signal_normalization=signal_normalization,
            allow_clipping_in_normalization=allow_clipping_in_normalization,
            symmetric_mels=symmetric_mels,
            use_lws=use_lws,
            frame_shift_ms=frame_shift_ms,
            max_abs_value=max_abs_value,
        )

    def _get_melspectrogram(self, audio: np.array):
        D = _stft(
            preemphasis(audio, self.preemphasis, self.preemphasize),
            n_fft=self.filter_length,
            win_size=self.win_length,
            hop_size=self.get_hop_size(),
            use_lws=self.use_lws,
        )
        S = (
            _amp_to_db(
                _linear_to_mel(
                    np.abs(D),
                    sampling_rate=self.sampling_rate,
                    n_fft=self.filter_length,
                    num_mels=self.n_mel_channels,
                    fmax=self.mel_fmax,
                    fmin=self.mel_fmin,
                )
            )
            - self.ref_level_db
        )
        if self.signal_normalization:
            return _normalize(
                S,
                max_abs_value=self.max_abs_value,
                min_level_db=self.min_level_db,
                allow_clipping_in_normalization=self.allow_clipping_in_normalization,
                symmetric_mels=self.symmetric_mels,
            )
        return S

    def __call__(self, audio: np.array):
        return self._get_melspectrogram(audio)

    def invert(
        mel_spectrogram: np.array,
        power,
    ):
        """Converts mel spectrogram to waveform using librosa"""
        if signal_normalization:
            D = _denormalize(
                mel_spectrogram,
                max_abs_value=self.max_abs_value,
                min_level_db=self.min_level_db,
                allow_clipping_in_normalization=self.allow_clipping_in_normalization,
                symmetric_mels=self.symmetric_mels,
            )
        else:
            D = mel_spectrogram

        S = _mel_to_linear(
            _db_to_amp(D + self.ref_level_db),
            sampling_rate=self.sampling_rate,
            n_fft=self.filter_length,
            num_mels=self.n_mel_channels,
            fmax=self.mel_fmax,
            fmin=self.mel_fmin,
        )  # Convert back to linear

        if use_lws:
            processor = _lws_processor()
            D = processor.run_lws(S.astype(np.float64).T ** power)
            y = processor.istft(D).astype(np.float32)
            return inv_preemphasis(y, self.preemphasis, self.preemphasize)
        else:
            return inv_preemphasis(
                _griffin_lim(S**power), self.preemphasis, self.preemphasize
            )
