import torch


def magnitude_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)

def normalize_mag_map(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (torch.amax(x) + eps)

def magnitude_nmse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:

    if pred.dim() == 2:
        num = torch.sum((pred - target) ** 2)
        den = torch.sum(target ** 2).clamp_min(eps)
        return num / den

    pred_flat = pred.reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)

    num = torch.sum((pred_flat - target_flat) ** 2, dim=1)
    den = torch.sum(target_flat ** 2, dim=1).clamp_min(eps)

    return torch.mean(num / den)

def weighted_nmse_loss(
    pred: torch.Tensor,
    target_n: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    pred:     predicted magnitude map, same scale convention as current code
    target_n: sample-wise normalized GT magnitude map
    """
    weights = 0.1 + target_n.clamp_min(0.0).pow(1.5)
    num = torch.sum(weights * (pred - target_n) ** 2)
    den = torch.sum(weights * (target_n ** 2)) + eps
    return num / den

# def topk_shape_loss(
#     pred_n: torch.Tensor,
#     target_n: torch.Tensor,
#     # topk_ratio: float = 0.125,
#     topk_ratio: float = 0.0625,
# ) -> torch.Tensor:
#     pred_flat = pred_n.reshape(-1)
#     target_flat = target_n.reshape(-1)

#     weight_power = 1.5
#     eps = 1e-8

#     k = max(1, int(round(topk_ratio * target_flat.numel())))
#     topk_idx = torch.topk(target_flat, k=k, largest=True).indices

#     pred_topk = pred_flat[topk_idx]
#     target_topk = target_flat[topk_idx]

#     weights = target_topk.clamp_min(eps).pow(weight_power)
#     weights = weights / weights.sum().clamp_min(eps)

#     return torch.sum(weights * (pred_topk - target_topk) ** 2)

def topk_shape_loss(
    pred_n: torch.Tensor,
    target_n: torch.Tensor,
    topk_ratio: float = 0.125,
) -> torch.Tensor:
    pred_flat = pred_n.reshape(-1)
    target_flat = target_n.reshape(-1)
    k = max(1, int(round(topk_ratio * target_flat.numel())))
    topk_idx = torch.topk(target_flat, k=k, largest=True).indices
    return torch.mean((pred_flat[topk_idx] - target_flat[topk_idx]) ** 2)

def weighted_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    weights = torch.ones_like(target)
    mask = target > 0.05
    weights[mask] = 1.0 + 20.0 * target[mask]

    abs_err = (pred - target).abs()
    return (weights * abs_err).sum() / weights.sum().clamp_min(eps)

def hybrid_magnitude_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    topk_ratio: float = 0.125,
    eps: float = 1e-8,
    return_terms: bool = False,
):
    pred_n = pred
    target_n = normalize_mag_map(target, eps=eps)
    # target_n = target

    abs_loss = magnitude_nmse_loss(pred_n, target_n, eps=eps)

    # abs_loss = weighted_nmse_loss(pred=pred_n,target_n=target_n,eps=eps)
    topk_loss = topk_shape_loss(pred_n, target_n, topk_ratio=topk_ratio)

    total_loss = 0.7 * abs_loss + 0.3 * topk_loss

    if return_terms:
        # target_power = torch.mean(target_n ** 2)
        return (
            total_loss,
            abs_loss.detach(),
            topk_loss.detach(),
            # target_power.detach(),
        )

    return total_loss