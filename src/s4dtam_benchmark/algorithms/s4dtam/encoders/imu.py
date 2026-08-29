import numpy as np

from .base import EncodedObservation, ModalityEncoder


class IMUEncoder(ModalityEncoder):
    modality = "imu"

    def encode(self, sample: np.ndarray, timestamp: float) -> EncodedObservation:
        return self._encode_numeric(sample, timestamp)
