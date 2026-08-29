import numpy as np

from .base import EncodedObservation, ModalityEncoder


class IMUEncoder(ModalityEncoder):
    modality = "imu"

    def encode(self, sample: np.ndarray, timestamp: float) -> EncodedObservation:
        measurement = self._validated_sample(sample).reshape(-1)
        if measurement.shape != (6,):
            raise ValueError("imu sample must contain [ax, ay, az, gx, gy, gz]")
        acceleration, angular_velocity = measurement[:3], measurement[3:]
        descriptor = np.concatenate(
            (
                acceleration,
                angular_velocity,
                [np.linalg.norm(acceleration), np.linalg.norm(angular_velocity)],
                np.cross(acceleration, angular_velocity),
            )
        )
        return self._project(descriptor, timestamp)
