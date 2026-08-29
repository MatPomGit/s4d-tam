import numpy as np

from .base import EncodedObservation, ModalityEncoder, distribution_descriptor


class RGBEncoder(ModalityEncoder):
    modality = "rgb"

    def encode(self, sample: np.ndarray, timestamp: float) -> EncodedObservation:
        image = self._validated_sample(sample)
        if image.ndim != 3 or image.shape[-1] not in (3, 4):
            raise ValueError("rgb sample must have shape (height, width, 3|4)")
        image = image[..., :3]
        channels = [distribution_descriptor(image[..., index]) for index in range(3)]
        horizontal = np.mean(np.abs(np.diff(image, axis=1))) if image.shape[1] > 1 else 0.0
        vertical = np.mean(np.abs(np.diff(image, axis=0))) if image.shape[0] > 1 else 0.0
        descriptor = np.concatenate((*channels, [horizontal, vertical]))
        return self._project(descriptor, timestamp)
