import numpy as np

from .base import EncodedObservation, ModalityEncoder, distribution_descriptor


class LiDAREncoder(ModalityEncoder):
    modality = "lidar"

    def encode(self, sample: np.ndarray, timestamp: float) -> EncodedObservation:
        points = self._validated_sample(sample)
        if points.ndim != 2 or points.shape[1] not in (3, 4):
            raise ValueError("lidar sample must have shape (points, 3|4)")
        xyz = points[:, :3]
        ranges = np.linalg.norm(xyz, axis=1)
        descriptor = [xyz.mean(axis=0), xyz.std(axis=0), distribution_descriptor(ranges)]
        if points.shape[1] == 4:
            descriptor.append(distribution_descriptor(points[:, 3]))
        else:
            descriptor.append(np.zeros(7))
        return self._project(np.concatenate(descriptor), timestamp)
