import os
import csv
import heapq
import random
from argparse import ArgumentParser

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from arguments import ModelParams, OptimizationParams, get_combined_args
from scene import Scene, GaussianModel
from gaussian_renderer import render

from gaussian_renderer.beam_subcarrier import render_beam_subcarrier


def prepare_dir(path: str):
    os.makedirs(path, exist_ok=True)


def add_colorbar(fig, ax, im, size="3.5%", pad=0.08):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=size, pad=pad)
    fig.colorbar(im, cax=cax)


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Render GT/pred/error maps from a saved checkpoint")
    mp = ModelParams(parser)
    op = OptimizationParams(parser)

    parser.add_argument("--checkpoint_path", type=str, default="outputs/20260414_024831/model.pth")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--num_samples", type=int, default=0,
                        help="0 means analyze the full split. Otherwise analyze this many samples.")
    parser.add_argument("--topk", type=int, default=20,
                        help="How many worst samples to save as detailed figures.")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--save_signed", action="store_true",
                        help="Also save mean signed error heatmap.")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device, e.g. cuda or cpu.")
    return parser


@torch.no_grad()
def render_sample(scene, gaussians, rx_pos):
    return render_beam_subcarrier(
        rx_pos=rx_pos,
        pc=gaussians,
        num_beams=scene.num_beams,
        num_subcarriers=scene.num_subcarriers,
        support_radius=getattr(scene, "plane_support_radius", 1),
    )


@torch.no_grad()
def evaluate_index(scene, gaussians, dataset, idx, device):
    magnitude, rx_pos = dataset[idx]
    gt = magnitude.to(device).reshape(scene.beam_rows, scene.beam_cols)
    rx_pos = rx_pos.to(device)
    out = render_sample(scene, gaussians, rx_pos)
    pred = out["render"]
    err = pred - gt
    abs_err = err.abs()

    mae = abs_err.mean().item()
    mse = (err ** 2).mean().item()
    gt_l1 = gt.abs().mean().item()
    gt_l2 = (gt ** 2).mean().item()
    rel_l1 = mae / max(gt_l1, 1e-12)
    rel_l2 = mse / max(gt_l2, 1e-12)

    return {
        "idx": idx,
        "gt": gt.detach().cpu(),
        "pred": pred.detach().cpu(),
        "err": err.detach().cpu(),
        "abs_err": abs_err.detach().cpu(),
        "mae": mae,
        "mse": mse,
        "rel_l1": rel_l1,
        "rel_l2": rel_l2,
    }


def save_metrics_csv(rows, path):
    fieldnames = ["idx", "mae", "mse", "rel_l1", "rel_l2"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fieldnames})


def save_summary_heatmaps(mean_abs_err, mean_signed_err, out_dir, save_signed=False):
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    im = ax.imshow(mean_abs_err, aspect="equal", interpolation="nearest")
    ax.set_title("Mean Absolute Error Map")
    add_colorbar(fig, ax, im)
    fig.savefig(os.path.join(out_dir, "mean_abs_error_map.png"), dpi=200)
    plt.close(fig)

    row_mean = mean_abs_err.mean(axis=1)
    col_mean = mean_abs_err.mean(axis=0)
    fig, axes = plt.subplots(2, 1, figsize=(8, 5), constrained_layout=True)
    axes[0].plot(row_mean)
    axes[0].set_title("Row-wise Mean Absolute Error")
    axes[0].set_xlabel("Row index")
    axes[0].set_ylabel("Mean abs err")
    axes[1].plot(col_mean)
    axes[1].set_title("Column-wise Mean Absolute Error")
    axes[1].set_xlabel("Column index")
    axes[1].set_ylabel("Mean abs err")
    fig.savefig(os.path.join(out_dir, "mean_abs_error_profiles.png"), dpi=200)
    plt.close(fig)

    if save_signed:
        vmax = float(np.max(np.abs(mean_signed_err)))
        norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax) if vmax > 0 else None
        fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
        im = ax.imshow(mean_signed_err, aspect="equal", interpolation="nearest", cmap="coolwarm", norm=norm)
        ax.set_title("Mean Signed Error Map (Pred - GT)")
        add_colorbar(fig, ax, im)
        fig.savefig(os.path.join(out_dir, "mean_signed_error_map.png"), dpi=200)
        plt.close(fig)


def save_sample_figure(result, rank, out_dir):
    gt = result["gt"].numpy()
    pred = result["pred"].numpy()
    err = result["err"].numpy()
    abs_err = result["abs_err"].numpy()

    vmax_main = float(max(gt.max(), pred.max()))
    vmax_abs = float(abs_err.max())
    vmax_signed = float(np.max(np.abs(err)))
    signed_norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax_signed, vmax=vmax_signed) if vmax_signed > 0 else None

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)

    im0 = axes[0, 0].imshow(gt, aspect="equal", interpolation="nearest", vmin=0.0, vmax=vmax_main)
    axes[0, 0].set_title("Ground Truth")
    add_colorbar(fig, axes[0, 0], im0)

    im1 = axes[0, 1].imshow(pred, aspect="equal", interpolation="nearest", vmin=0.0, vmax=vmax_main)
    axes[0, 1].set_title("Predicted")
    add_colorbar(fig, axes[0, 1], im1)

    im2 = axes[1, 0].imshow(abs_err, aspect="equal", interpolation="nearest", vmin=0.0, vmax=vmax_abs)
    axes[1, 0].set_title("Absolute Error")
    add_colorbar(fig, axes[1, 0], im2)

    im3 = axes[1, 1].imshow(err, aspect="equal", interpolation="nearest", cmap="coolwarm", norm=signed_norm)
    axes[1, 1].set_title("Signed Error (Pred - GT)")
    add_colorbar(fig, axes[1, 1], im3)

    for ax in axes.ravel():
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_aspect("equal")

    fig.suptitle(
        f"rank={rank:02d}  idx={result['idx']}  MAE={result['mae']:.6f}  relL1={result['rel_l1']:.4f}",
        fontsize=12,
    )
    fig.savefig(os.path.join(out_dir, f"worst_{rank:02d}_idx_{result['idx']}.png"), dpi=220)
    plt.close(fig)


def main():
    parser = build_parser()
    args = get_combined_args(parser)

    checkpoint_path = os.path.abspath(args.checkpoint_path)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model_dir = os.path.dirname(checkpoint_path)
    if getattr(args, "model_path", None) in (None, "", "outputs"):
        args.model_path = model_dir
    else:
        args.model_path = model_dir

    requested_device = getattr(args, "device", None)
    data_device = getattr(args, "data_device", "cuda")

    if requested_device is not None:
        device = torch.device(requested_device)
    else:
        device = torch.device(data_device if torch.cuda.is_available() else "cpu")
        
    gaussians = GaussianModel(
        target_gaussians=12000,
        optimizer_type=args.optimizer_type,
        device=str(device),
        init_range=1,
        num_beams=getattr(args, "num_beams", 100),
        num_subcarriers=getattr(args, "num_subcarriers", 100),
        plane_init_sigma_beam=getattr(args, "plane_init_sigma_beam", 0.70),
        plane_init_sigma_subcarrier=getattr(args, "plane_init_sigma_subcarrier", 0.70),
        plane_min_sigma=getattr(args, "plane_min_sigma", 0.25),
        plane_max_sigma=getattr(args, "plane_max_sigma", 1.20),
    )

    scene = Scene(args, gaussians)

    ckpt = torch.load(checkpoint_path, map_location=device)

    def unwrap_gaussian_payload(obj):
        valid_lengths = {11, 17}

        if isinstance(obj, (tuple, list)) and len(obj) in valid_lengths:
            return obj

        if isinstance(obj, (tuple, list)) and len(obj) > 0:
            first = obj[0]
            if isinstance(first, (tuple, list)) and len(first) in valid_lengths:
                return first

        if isinstance(obj, dict):
            candidate_keys = [
                "gaussians",
                "gaussian_model",
                "gaussian_state",
                "model",
                "model_state",
                "capture",
                "state",
                "params",
            ]
            for k in candidate_keys:
                if k in obj:
                    v = obj[k]
                    if isinstance(v, (tuple, list)) and len(v) in valid_lengths:
                        return v
                    if isinstance(v, (tuple, list)) and len(v) > 0:
                        first = v[0]
                        if isinstance(first, (tuple, list)) and len(first) in valid_lengths:
                            return first

        raise ValueError(
            f"Could not find Gaussian payload for restore. "
            f"Top-level type={type(obj)}, "
            f"keys={list(obj.keys()) if isinstance(obj, dict) else 'N/A'}"
        )

    gaussian_payload = unwrap_gaussian_payload(ckpt)
    gaussians.restore(gaussian_payload, args)

    gaussians.dynamic_gain_net.eval()

    dataset = scene.test_set if args.split == "test" else scene.train_set

    total = len(dataset)
    if args.num_samples and args.num_samples > 0:
        rng = random.Random(args.seed)
        indices = rng.sample(range(total), min(args.num_samples, total))
    else:
        indices = list(range(total))

    out_dir = os.path.join(model_dir, f"error_maps_{args.split}")
    prepare_dir(out_dir)

    print(f"[ErrorMap] checkpoint: {checkpoint_path}")
    print(f"[ErrorMap] split: {args.split}")
    print(f"[ErrorMap] samples: {len(indices)} / {total}")
    print(f"[ErrorMap] save dir: {out_dir}")

    rows = []
    mean_abs_acc = None
    mean_signed_acc = None
    worst_heap = []  # min-heap by MAE

    for count, idx in enumerate(indices, start=1):
        result = evaluate_index(scene, gaussians, dataset, idx, device)
        rows.append({
            "idx": result["idx"],
            "mae": result["mae"],
            "mse": result["mse"],
            "rel_l1": result["rel_l1"],
            "rel_l2": result["rel_l2"],
        })

        abs_np = result["abs_err"].numpy()
        err_np = result["err"].numpy()

        if mean_abs_acc is None:
            mean_abs_acc = np.zeros_like(abs_np, dtype=np.float64)
            mean_signed_acc = np.zeros_like(err_np, dtype=np.float64)
        mean_abs_acc += abs_np
        mean_signed_acc += err_np

        key = result["mae"]
        packed = (key, result)
        if len(worst_heap) < args.topk:
            heapq.heappush(worst_heap, packed)
        else:
            if key > worst_heap[0][0]:
                heapq.heapreplace(worst_heap, packed)

        if count % 50 == 0 or count == len(indices):
            print(f"[ErrorMap] processed {count}/{len(indices)}")

    rows_sorted = sorted(rows, key=lambda x: x["mae"], reverse=True)
    save_metrics_csv(rows_sorted, os.path.join(out_dir, "sample_metrics.csv"))

    mean_abs_err = mean_abs_acc / len(indices)
    mean_signed_err = mean_signed_acc / len(indices)
    save_summary_heatmaps(mean_abs_err, mean_signed_err, out_dir, save_signed=args.save_signed)

    worst_results = [item[1] for item in sorted(worst_heap, key=lambda x: x[0], reverse=True)]
    for rank, result in enumerate(worst_results, start=1):
        save_sample_figure(result, rank, out_dir)

    maes = np.array([r["mae"] for r in rows], dtype=np.float64)
    rels = np.array([r["rel_l1"] for r in rows], dtype=np.float64)
    print(f"[ErrorMap] mean MAE: {maes.mean():.6f}")
    print(f"[ErrorMap] median MAE: {np.median(maes):.6f}")
    print(f"[ErrorMap] mean relL1: {rels.mean():.6f}")
    print(f"[ErrorMap] median relL1: {np.median(rels):.6f}")
    print("[ErrorMap] done.")


if __name__ == "__main__":
    main()
