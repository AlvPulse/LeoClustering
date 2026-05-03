import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from src.embeddings.base import BaseEmbeddingExtractor

class YAMNetExtractor(BaseEmbeddingExtractor):
    def __init__(self):
        # YAMNet expects 16kHz
        self._target_sr = 16000
        self._dim = 1024

        # Ensure we use CPU for bitwise determinism if requested, or just disable
        # non-deterministic TF ops. YAMNet on CPU is extremely fast and deterministic.
        tf.config.set_visible_devices([], 'GPU')

        # Load model from TF Hub
        self.model = hub.load('https://tfhub.dev/google/yamnet/1')

    @property
    def target_sr(self) -> int:
        return self._target_sr

    @property
    def dimension(self) -> int:
        return self._dim

    def extract(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if len(audio) == 0:
            return np.full(self.dimension, np.nan)

        y = self.resample(audio, sr)

        # YAMNet expects [-1.0, +1.0] float32 arrays
        # The model automatically frames into 0.96s windows with 0.48s hop
        _, embeddings, _ = self.model(y)

        # embeddings shape is (num_frames, 1024)
        # Average across windows
        embeddings = embeddings.numpy()
        if len(embeddings) == 0:
            return np.zeros(self.dimension, dtype=np.float32)

        return np.mean(embeddings, axis=0)
