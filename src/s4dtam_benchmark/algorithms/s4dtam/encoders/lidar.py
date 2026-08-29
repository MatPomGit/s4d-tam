import numpy as np

from .base import EncodedObservation, ModalityEncoder


class LiDAREncoder(ModalityEncoder):
    modality = "lidar"

    def encode(self, sample: np.ndarray, timestamp: float) -> EncodedObservation:
        return self._encode_numeric(sample, timestamp)
