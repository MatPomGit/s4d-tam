import numpy as np

from .base import EncodedObservation, ModalityEncoder, distribution_descriptor


class ThermalEncoder(ModalityEncoder):
    modality = "thermal"

    def encode(self, sample: np.ndarray, timestamp: float) -> EncodedObservation:
        image = self._validated_sample(sample)
        if image.ndim == 3 and image.shape[-1] == 1:
            image = image[..., 0]
        if image.ndim != 2:
            raise ValueError("thermal sample must have shape (height, width) or (height, width, 1)")
        horizontal = np.mean(np.abs(np.diff(image, axis=1))) if image.shape[1] > 1 else 0.0
        vertical = np.mean(np.abs(np.diff(image, axis=0))) if image.shape[0] > 1 else 0.0
        descriptor = np.concatenate((distribution_descriptor(image), [horizontal, vertical]))
        return self._project(descriptor, timestamp)
