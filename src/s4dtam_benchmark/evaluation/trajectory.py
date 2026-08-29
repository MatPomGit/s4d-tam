from __future__ import annotations

import numpy as np


def align_se3(reference: np.ndarray, estimate: np.ndarray) -> np.ndarray:
    """Rigid Umeyama alignment without scale (SE(3), not Sim(3))."""
    if reference.shape != estimate.shape or reference.ndim != 2 or reference.shape[1] != 3:
        raise ValueError("reference and estimate must both have shape [N, 3]")
    ref_mean, est_mean = reference.mean(axis=0), estimate.mean(axis=0)
    ref_centered, est_centered = reference - ref_mean, estimate - est_mean
    covariance = est_centered.T @ ref_centered / len(reference)
    u, _, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ correction @ u.T
    translation = ref_mean - rotation @ est_mean
    return (rotation @ estimate.T).T + translation


def trajectory_metrics(
    reference: np.ndarray, estimate: np.ndarray, delta_frames: int = 1, align: bool = True
) -> dict[str, float]:
    estimate_eval = align_se3(reference, estimate) if align else estimate
    errors = np.linalg.norm(estimate_eval - reference, axis=1)
    if len(reference) <= delta_frames:
        relative = np.array([], dtype=float)
    else:
        ref_delta = reference[delta_frames:] - reference[:-delta_frames]
        est_delta = estimate_eval[delta_frames:] - estimate_eval[:-delta_frames]
        relative = np.linalg.norm(est_delta - ref_delta, axis=1)
    path_length = float(np.linalg.norm(np.diff(reference, axis=0), axis=1).sum())
    return {
        "trajectory/ate_rmse_m": float(np.sqrt(np.mean(errors**2))),
        "trajectory/ate_median_m": float(np.median(errors)),
        "trajectory/ate_p95_m": float(np.quantile(errors, 0.95)),
        "trajectory/rpe_translation_rmse_m": (
            float(np.sqrt(np.mean(relative**2))) if relative.size else float("nan")
        ),
        "trajectory/final_drift_m": float(errors[-1]),
        "trajectory/final_drift_percent": 100.0 * float(errors[-1]) / max(path_length, 1e-12),
        "trajectory/path_length_m": path_length,
    }


def rotation_rpe_deg(
    reference_xyzw: np.ndarray, estimate_xyzw: np.ndarray, delta_frames: int = 1
) -> float:
    """RMSE of relative quaternion angle; quaternion convention is [x,y,z,w]."""
    if reference_xyzw.shape != estimate_xyzw.shape or reference_xyzw.shape[1] != 4:
        raise ValueError("quaternions must have matching [N,4] shape")

    def normalize(q: np.ndarray) -> np.ndarray:
        return q / np.linalg.norm(q, axis=1, keepdims=True)

    def conjugate(q: np.ndarray) -> np.ndarray:
        result = q.copy()
        result[:, :3] *= -1
        return result

    def multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        av, aw, bv, bw = a[:, :3], a[:, 3:], b[:, :3], b[:, 3:]
        vector = aw * bv + bw * av + np.cross(av, bv)
        scalar = aw * bw - np.sum(av * bv, axis=1, keepdims=True)
        return np.hstack((vector, scalar))

    reference, estimate = normalize(reference_xyzw), normalize(estimate_xyzw)
    ref_delta = multiply(conjugate(reference[:-delta_frames]), reference[delta_frames:])
    est_delta = multiply(conjugate(estimate[:-delta_frames]), estimate[delta_frames:])
    error = multiply(conjugate(ref_delta), est_delta)
    angles = 2.0 * np.arccos(np.clip(np.abs(normalize(error)[:, 3]), 0.0, 1.0))
    return float(np.degrees(np.sqrt(np.mean(angles**2))))
