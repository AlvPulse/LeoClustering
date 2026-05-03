from abc import ABC, abstractmethod
import numpy as np
import librosa
import warnings

class BaseEmbeddingExtractor(ABC):
    @property
    @abstractmethod
    def target_sr(self) -> int:
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        pass

    def resample(self, audio: np.ndarray, orig_sr: int) -> np.ndarray:
        if orig_sr == self.target_sr:
            return audio
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=self.target_sr, res_type='soxr_hq')

    @abstractmethod
    def extract(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Takes raw audio and its sample rate.
        Returns a 1D numpy array of shape (dimension,).
        """
        pass
