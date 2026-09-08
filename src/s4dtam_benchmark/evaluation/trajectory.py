from __future__ import annotations

import numpy as np


def _validate_positions(reference: np.ndarray, estimate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    if reference.shape != estimate.shape or reference.ndim != 2 or reference.shape[1] != 3:
        raise ValueError("reference and estimate must both have shape [N, 3]")
    if len(reference) == 0 or not np.all(np.isfinite(reference)) or not np.all(np.isfinite(estimate)):
        raise ValueError("reference and estimate must be non-empty and finite")
    return reference, estimate


def align_se3(reference: np.ndarray, estimate: np.ndarray) -> np.ndarray:
    """Rigid Umeyama alignment without scale (SE(3))."""
    reference, estimate = _validate_positions(reference, estimate)
    ref_mean, est_mean = reference.mean(axis=0), estimate.mean(axis=0)
    ref_centered, est_centered = reference - ref_mean, estimate - est_mean
    covariance = ref_centered.T @ est_centered / len(reference)
    u, _, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1.0
    rotation = u @ correction @ vt
    translation = ref_mean - rotation @ est_mean
    return (rotation @ estimate.T).T + translation


def align_sim3(reference: np.ndarray, estimate: np.ndarray) -> tuple[np.ndarray, float]:
    """Similarity Umeyama alignment mapping estimate onto reference."""
    reference, estimate = _validate_positions(reference, estimate)
    if len(reference) < 2:
        raise ValueError("Sim(3) alignment requires at least two samples")
    ref_mean, est_mean = reference.mean(axis=0), estimate.mean(axis=0)
    ref_centered, est_centered = reference - ref_mean, estimate - est_mean
    variance = float(np.sum(est_centered**2) / len(estimate))
    if variance <= np.finfo(float).eps:
        raise ValueError("Sim(3) alignment requires non-degenerate estimated positions")
    covariance = ref_centered.T @ est_centered / len(reference)
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1.0
    rotation = u @ correction @ vt
    scale = float(np.sum(singular * np.diag(correction)) / variance)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Sim(3) alignment produced a non-positive scale")
    translation = ref_mean - scale * (rotation @ est_mean)
    aligned = scale * (rotation @ estimate.T).T + translation
    return aligned, scale


def trajectory_metrics(
    reference: np.ndarray,
    estimate: np.ndarray,
    delta_frames: int = 1,
    align: bool = True,
    alignment_mode: str = "se3",
) -> dict[str, float]:
    reference, estimate = _validate_positions(reference, estimate)
    if alignment_mode not in {"se3", "sim3", "none"}:
        raise ValueError("alignment_mode must be se3, sim3 or none")
    scale = 1.0
    if not align or alignment_mode == "none":
        estimate_eval = estimate
    elif alignment_mode == "se3":
        estimate_eval = align_se3(reference, estimate)
    else:
        estimate_eval, scale = align_sim3(reference, estimate)
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
        "trajectory/alignment_scale": scale,
    }


def rotation_rpe_deg(
    reference_xyzw: np.ndarray,
    estimate_xyzw: np.ndarray,
    delta_frames: int = 1,
    valid_mask: np.ndarray | None = None,
) -> float:
    """RMSE of relative quaternion angle; quaternion convention is [x,y,z,w]."""
    reference_xyzw = np.asarray(reference_xyzw, dtype=float)
    estimate_xyzw = np.asarray(estimate_xyzw, dtype=float)
    if (
        reference_xyzw.shape != estimate_xyzw.shape
        or reference_xyzw.ndim != 2
        or reference_xyzw.shape[1] != 4
    ):
        raise ValueError("quaternions must have matching [N,4] shape")
    if len(reference_xyzw) <= delta_frames:
        return float("nan")

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
    if valid_mask is not None:
        valid = np.asarray(valid_mask)
        if valid.dtype != np.bool_ or valid.shape != (len(reference_xyzw),):
            raise ValueError("valid_mask must be boolean with shape [N]")
        pair_valid = valid[:-delta_frames] & valid[delta_frames:]
        angles = angles[pair_valid]
    if angles.size == 0:
        return float("nan")
    return float(np.degrees(np.sqrt(np.mean(angles**2))))
