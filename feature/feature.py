# Copyright 2023 Sony Group Corporation.

import numpy as np
import librosa


class SpectralFeature(object):
    def __init__(self, wav=None, fft_size=None, stft_hop_size=None, center=None, config=None):
        self._config = config

        self._wav_ch = wav.shape[1]
        wav_c_contiguous_example = np.require(wav[:, 0], dtype=np.float32, requirements=['C'])
        spec_example = librosa.core.stft(wav_c_contiguous_example,
                                         n_fft=fft_size,
                                         hop_length=stft_hop_size,
                                         center=center)  # used for num_frame
        self._num_bin = int(fft_size / 2) + 1
        self._num_frame = spec_example.shape[1]
        self._complex_spec = np.ones((self._wav_ch, self._num_bin, self._num_frame), dtype='complex64')

        self._complex_spec[0] = spec_example
        for i in range(1, self._wav_ch):
            wav_c_contiguous = np.require(wav[:, i], dtype=np.float32, requirements=['C'])
            self._complex_spec[i] = librosa.core.stft(wav_c_contiguous,
                                                      n_fft=fft_size,
                                                      hop_length=stft_hop_size,
                                                      center=center)

    def amplitude(self):
        amp = np.zeros((self._wav_ch, self._num_bin, self._num_frame), dtype='float32')
        for i in range(self._wav_ch):
            amp[i] = np.abs(self._complex_spec[i])
        return amp

    def phasediff(self):
        phasediff = np.zeros((self._wav_ch - 1, self._num_bin, self._num_frame), dtype='float32')
        wav_ch_base = self._config["base_channel"]
        spec_angle_base = np.angle(self._complex_spec[wav_ch_base])
        ch_wo_base = np.delete(np.arange(self._wav_ch), wav_ch_base)
        for enu_i, i in enumerate(ch_wo_base):
            spec_angle = np.angle(self._complex_spec[i]) - spec_angle_base
            spec_angle[spec_angle < 0] += 2 * np.pi
            phasediff[enu_i] = spec_angle
        return phasediff
