import numpy as np
import torch
from panns_inference import AudioTagging
from src.embeddings.base import BaseEmbeddingExtractor

class PANNsExtractor(BaseEmbeddingExtractor):
    def __init__(self):
        # PANNs expects 32kHz (wait, let's check standard panns_inference)
        # Default AudioTagging uses 32kHz. We will use 32000.
        self._target_sr = 32000
        self._dim = 2048

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self.model = AudioTagging(checkpoint_path=None, device=device)
        self.device = device

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

        # Expand dims for batch = 1
        y_batch = y[None, :]

        # AudioTagging returns (clipwise_output, embedding)
        _, embedding = self.model.inference(y_batch)

        return embedding[0]
