from typing import Dict
import torch
from scene.gaussian_model import GaussianModel

def render_beam_subcarrier(
    rx_pos: torch.Tensor,
    pc: GaussianModel,
    num_beams: int,
    num_subcarriers: int,
    support_radius: int = 1,
) -> Dict[str, torch.Tensor]:
    rx_pos = rx_pos.view(1, 3).to(pc.device, dtype=pc.get_plane_center.dtype)

    gains = pc.get_dynamic_gain_weight(rx_pos).squeeze(-1)      # (N,)
    centers = pc.get_plane_center                               # (N,2) [beam, subcarrier]
    sigmas = pc.get_plane_sigma                                 # (N,2)

    device = centers.device
    dtype = centers.dtype

    offs = torch.arange(-support_radius, support_radius + 1, device=device)
    db, dk = torch.meshgrid(offs, offs, indexing="xy")
    db = db.reshape(-1)                                         # (P,)
    dk = dk.reshape(-1)                                         # (P,)

    b0 = centers[:, 0].round().long().unsqueeze(1)
    k0 = centers[:, 1].round().long().unsqueeze(1)

    b_idx = b0 + db.unsqueeze(0)                                # (N,P)
    k_idx = k0 + dk.unsqueeze(0)                                # (N,P)

    valid = (
        (b_idx >= 0) & (b_idx < num_beams) &
        (k_idx >= 0) & (k_idx < num_subcarriers)
    )

    diff_b = b_idx.to(dtype) - centers[:, 0:1]
    diff_k = k_idx.to(dtype) - centers[:, 1:2]

    sigma_b = sigmas[:, 0:1]
    sigma_k = sigmas[:, 1:2]

    w = torch.exp(
        -0.5 * (diff_b / sigma_b).pow(2)
        -0.5 * (diff_k / sigma_k).pow(2)
    )
    w = w * valid.to(dtype)
    w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-12)

    contrib = gains[:, None] * w                                # (N,P)

    flat_idx = (k_idx * num_beams + b_idx)
    render_flat = torch.zeros(num_subcarriers * num_beams, device=device, dtype=dtype)

    render_flat.index_add_(
        0,
        flat_idx[valid].reshape(-1),
        contrib[valid].reshape(-1),
    )

    render = render_flat.view(num_subcarriers, num_beams)       # row=subcarrier, col=beam

    return {
        "render": render,
        "magnitude": render,
        "per_gaussian_importance": gains.abs(),
        "gain_weight": gains,
        "plane_centers": centers,
        "plane_sigmas": sigmas,
    }