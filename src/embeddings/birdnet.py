import numpy as np
import tensorflow as tf
import os
import librosa
from src.embeddings.base import BaseEmbeddingExtractor

class BirdNetExtractor(BaseEmbeddingExtractor):
    def __init__(self):
        self._target_sr = 48000
        tf.config.set_visible_devices([], 'GPU')

        from birdnetlib.analyzer import Analyzer
        analyzer = Analyzer()
        self.model_path = analyzer.model_path

        self.interpreter = tf.lite.Interpreter(model_path=self.model_path, experimental_preserve_all_tensors=True)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()[0]
        tensor_details = self.interpreter.get_tensor_details()
        self.embedding_idx = None

        for t in tensor_details:
            if 'flatten' in t['name'].lower() or 'global_average_pooling2d' in t['name'].lower() or 'post/melspec_dense' in t['name'].lower() or 'post/dense' in t['name'].lower():
                self.embedding_idx = t['index']
                self._dim = t['shape'][-1]

        if self.embedding_idx is None:
            final_layer_input_idx = tensor_details[-2]['index']
            self.embedding_idx = final_layer_input_idx
            self._dim = tensor_details[-2]['shape'][-1]

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

        chunk_size = int(3.0 * self._target_sr)

        if len(y) < chunk_size:
            y = np.pad(y, (0, chunk_size - len(y)))

        chunks = []
        hop_length = int(1.5 * self._target_sr)
        for i in range(0, len(y) - chunk_size + 1, hop_length):
            chunks.append(y[i:i + chunk_size])

        if not chunks:
            chunks.append(np.pad(y, (0, chunk_size - len(y))))

        embeddings = []
        for chunk in chunks:
            input_data = np.expand_dims(chunk, axis=0).astype(np.float32)
            self.interpreter.set_tensor(self.input_details['index'], input_data)
            self.interpreter.invoke()
            emb = self.interpreter.get_tensor(self.embedding_idx)
            embeddings.append(np.copy(emb).flatten())

        return np.mean(embeddings, axis=0)
