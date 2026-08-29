import numpy as np

from .base import EncodedObservation, ModalityEncoder


class RGBEncoder(ModalityEncoder):
    modality = "rgb"

    def encode(self, sample: np.ndarray, timestamp: float) -> EncodedObservation:
        return self._encode_numeric(sample, timestamp)
