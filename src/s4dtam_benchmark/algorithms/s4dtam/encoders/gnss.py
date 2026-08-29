import numpy as np

from .base import EncodedObservation, ModalityEncoder


class GNSSEncoder(ModalityEncoder):
    modality = "gnss"

    def encode(self, sample: np.ndarray, timestamp: float) -> EncodedObservation:
        measurement = self._validated_sample(sample).reshape(-1)
        if measurement.size not in (3, 6):
            raise ValueError("gnss sample must contain position or position and uncertainty")
        position = measurement[:3]
        uncertainty = measurement[3:] if measurement.size == 6 else np.zeros(3)
        if np.any(uncertainty < 0):
            raise ValueError("gnss uncertainty cannot be negative")
        descriptor = np.concatenate((position, uncertainty, [np.linalg.norm(position)]))
        return self._project(descriptor, timestamp)
