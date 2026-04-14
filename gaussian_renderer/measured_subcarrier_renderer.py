from typing import Dict, Tuple

import torch

from scene.gaussian_model import GaussianModel
from . import (
    _ensure_pos_shape,
    _build_beam_uv_grid,
    _projected_angular_covariance,
    _gaussian_beam_weights_from_uv,
)


def render_measured_subcarrier_beam(
    rx_pos: torch.Tensor,
    tx_pos: torch.Tensor,
    pc: GaussianModel,
    num_subcarriers: int,
    beam_shape: Tuple[int, int] = (10, 10),
    scaling_modifier: float = 1.0,
    normalize_beam_weights: bool = True,
    covariance_floor: float = 1e-4,
    weight_floor: float = 0.0,
    output_layout: str = "beam_subcarrier",
) -> Dict[str, torch.Tensor]:
    rx_pos = _ensure_pos_shape(rx_pos).to(pc.get_xyz.device, dtype=pc.get_xyz.dtype)
    tx_pos = _ensure_pos_shape(tx_pos).to(pc.get_xyz.device, dtype=pc.get_xyz.dtype)

    means = pc.get_xyz
    covariances = pc.get_covariance(scaling_modifier)
    gain_weight = pc.get_dynamic_gain_weight(rx_pos)
    spectral_profile = pc.get_dynamic_spectral_profile(rx_pos, num_subcarriers=num_subcarriers)

    beam_centers_uv = _build_beam_uv_grid(
        horizontal=beam_shape[0],
        vertical=beam_shape[1],
        device=means.device,
        dtype=means.dtype,
    )

    beam_uv_mean, beam_cov_uv, _ = _projected_angular_covariance(
        means=means,
        covariances=covariances,
        array_pos=tx_pos,
        covariance_floor=covariance_floor,
    )

    beam_weights = _gaussian_beam_weights_from_uv(
        uv_mean=beam_uv_mean,
        cov_uv=beam_cov_uv,
        beam_centers_uv=beam_centers_uv,
        normalize=normalize_beam_weights,
        weight_floor=weight_floor,
        eig_floor=max(covariance_floor, 1e-4),
    )

    contributions = (
        gain_weight[:, :, None]
        * spectral_profile[:, :, None]
        * beam_weights[:, None, :]
    )
    H_kb = contributions.sum(dim=0)
    per_gaussian_importance = contributions.abs().sum(dim=(1, 2))

    if output_layout == "beam_subcarrier":
        render = H_kb.transpose(0, 1).contiguous()
    elif output_layout == "subcarrier_beam":
        render = H_kb
    else:
        raise ValueError(f"Unknown output_layout: {output_layout}")

    return {
        "render": render,
        "magnitude": render,
        "spectral_profile": spectral_profile,
        "beam_weights": beam_weights,
        "per_gaussian_importance": per_gaussian_importance,
        "beam_contributions": contributions,
        "gain_weight": gain_weight,
    }
