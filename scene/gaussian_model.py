import math
import os

import numpy as np
import scipy.io as sio
import torch
from torch import nn
import torch.nn.functional as F
from plyfile import PlyData, PlyElement

from utils.general_utils import (
    inverse_sigmoid,
    inverse_softplus,
    get_expon_lr_func,
    build_covariance_from_scaling_rotation,
    mkdir_p,
)

from typing import Optional, Dict, Any

class FourierFeatures(nn.Module):
    def __init__(self, in_dim=3, num_frequencies=6, include_input=True):
        super().__init__()
        self.in_dim = in_dim
        self.num_frequencies = num_frequencies
        self.include_input = include_input
        self.out_dim = in_dim * ((1 if include_input else 0) + 2 * num_frequencies)

        freq_bands = (2.0 ** torch.arange(num_frequencies, dtype=torch.float32)) * math.pi
        self.register_buffer("freq_bands", freq_bands, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., in_dim)
        [x, sin(f0*x), cos(f0*x), sin(f1*x), cos(f1*x), ...]
        """
        if self.num_frequencies == 0:
            if self.include_input:
                return x
            return x.new_empty(*x.shape[:-1], 0)

        freq_bands = self.freq_bands.to(device=x.device, dtype=x.dtype)

        # (..., F, D)
        x_proj = x.unsqueeze(-2) * freq_bands.view(*([1] * (x.dim() - 1)), -1, 1)

        sin_part = torch.sin(x_proj)
        cos_part = torch.cos(x_proj)

        # (..., F, 2, D) -> (..., 2*F*D)
        fourier = torch.stack((sin_part, cos_part), dim=-2).reshape(*x.shape[:-1], -1)

        if self.include_input:
            return torch.cat([x, fourier], dim=-1)
        return fourier

class DynamicGainNet(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 64,
        init_gain: float = 0.1,
        num_frequencies: int = 6,
        include_input: bool = True,
    ):
        super().__init__()

        self.pe = FourierFeatures(
            in_dim=3,
            num_frequencies=num_frequencies,
            include_input=include_input,
        )
        
        pe_dim = self.pe.out_dim
        mlp_in_dim = pe_dim * 3 + 1

        self.net = nn.Sequential(
            nn.Linear(mlp_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        nn.init.zeros_(self.net[-1].weight)
        init_bias = float(inverse_softplus(torch.tensor(init_gain)))
        nn.init.constant_(self.net[-1].bias, init_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (N, 10)
        xyz = x[:, 0:3]
        rx  = x[:, 3:6]
        rel = x[:, 6:9]
        logd = x[:, 9:10]

        feat = torch.cat(
            [
                self.pe(xyz),
                self.pe(rx),
                self.pe(rel),
                logd,
            ],
            dim=-1,
        )
        return self.net(feat)

class DynamicGainNet2D(nn.Module):
    def __init__(self, hidden_dim: int = 64, num_frequencies: int = 6, init_gain: float = 1.0):
        super().__init__()
        self.plane_pe = FourierFeatures(in_dim=2, num_frequencies=num_frequencies, include_input=True)
        self.rx_pe = FourierFeatures(in_dim=3, num_frequencies=num_frequencies, include_input=True)

        in_dim = self.plane_pe.out_dim + self.rx_pe.out_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        nn.init.zeros_(self.net[-1].weight)
        init_bias = float(inverse_softplus(torch.tensor(init_gain)))
        nn.init.constant_(self.net[-1].bias, init_bias)

    def forward(self, plane_coord_norm: torch.Tensor, rx_pos: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([self.plane_pe(plane_coord_norm), self.rx_pe(rx_pos)], dim=-1)
        return self.net(feat)

class DynamicCenterNet2D(nn.Module):
    def __init__(self, hidden_dim: int = 64, num_frequencies: int = 6):
        super().__init__()
        self.plane_pe = FourierFeatures(in_dim=2, num_frequencies=num_frequencies, include_input=True)
        self.rx_pe = FourierFeatures(in_dim=3, num_frequencies=num_frequencies, include_input=True)

        in_dim = self.plane_pe.out_dim + self.rx_pe.out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, plane_coord_norm: torch.Tensor, rx_pos: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([self.plane_pe(plane_coord_norm), self.rx_pe(rx_pos)], dim=-1)
        return self.net(feat)

class DynamicSigmaNet2D(nn.Module):
    def __init__(self, hidden_dim: int = 64, num_frequencies: int = 6):
        super().__init__()
        self.plane_pe = FourierFeatures(in_dim=2, num_frequencies=num_frequencies, include_input=True)
        self.rx_pe = FourierFeatures(in_dim=3, num_frequencies=num_frequencies, include_input=True)

        in_dim = self.plane_pe.out_dim + self.rx_pe.out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, plane_coord_norm: torch.Tensor, rx_pos: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([self.plane_pe(plane_coord_norm), self.rx_pe(rx_pos)], dim=-1)
        return self.net(feat)

class DynamicSpectralNet(nn.Module):
    def __init__(
        self,
        out_dim: int = 16,
        hidden_dim: int = 64,
        num_frequencies: int = 6,
        include_input: bool = True,
    ):
        super().__init__()

        self.pe = FourierFeatures(
            in_dim=3,
            num_frequencies=num_frequencies,
            include_input=include_input,
        )

        pe_dim = self.pe.out_dim
        mlp_in_dim = pe_dim * 3 + 1

        self.net = nn.Sequential(
            nn.Linear(mlp_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xyz = x[:, 0:3]
        rx  = x[:, 3:6]
        rel = x[:, 6:9]
        logd = x[:, 9:10]

        feat = torch.cat(
            [
                self.pe(xyz),
                self.pe(rx),
                self.pe(rel),
                logd,
            ],
            dim=-1,
        )
        return self.net(feat)


class GaussianModel:
    """MIMOGS Gaussian scene model

    Learnable attributes per Gaussian:
    - mean                  : xyz
    - covariance            : rotation + scaling
    - opacity-like weight   :opacity
    """

    def __init__(
        self,
        target_gaussians: int = 50_000,
        optimizer_type: str = "default",
        device: str = "cuda",
        init_range: float = 5.0,
        num_beams: int = 100,
        num_subcarriers: int = 100,
        plane_init_sigma_beam: float = 0.70,
        plane_init_sigma_subcarrier: float = 0.70,
        plane_min_sigma: float = 0.25,
        plane_max_sigma: float = 1.20,
        use_dynamic_center: bool = True,
        use_dynamic_sigma: bool = True,
        center_shift_max_beam: float = 1.5,
        center_shift_max_subcarrier: float = 1.5,
        sigma_log_shift_max_beam: float = 0.5,
        sigma_log_shift_max_subcarrier: float = 0.5,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.optimizer_type = optimizer_type
        self.target_gaussians = target_gaussians
        self.init_range = init_range

        self.num_beams = num_beams
        self.num_subcarriers = num_subcarriers
        self.plane_init_sigma_beam = plane_init_sigma_beam
        self.plane_init_sigma_subcarrier = plane_init_sigma_subcarrier
        self.plane_min_sigma = plane_min_sigma
        self.plane_max_sigma = plane_max_sigma
        self.use_dynamic_center = use_dynamic_center
        self.use_dynamic_sigma = use_dynamic_sigma
        self.center_shift_max_beam = center_shift_max_beam
        self.center_shift_max_subcarrier = center_shift_max_subcarrier
        self.sigma_log_shift_max_beam = sigma_log_shift_max_beam
        self.sigma_log_shift_max_subcarrier = sigma_log_shift_max_subcarrier

        self._plane_center = torch.empty(0, 2, device=self.device)
        self._plane_log_sigma = torch.empty(0, 2, device=self.device)
        self._opacity = torch.empty(0, 1, device=self.device)

        self.optimizer = None
        self.dynamic_gain_optimizer = None
        self.dynamic_center_optimizer = None
        self.dynamic_sigma_optimizer = None

        self.dynamic_gain_net = DynamicGainNet2D().to(self.device)
        self.dynamic_center_net = DynamicCenterNet2D().to(self.device)
        self.dynamic_sigma_net = DynamicSigmaNet2D().to(self.device)
        self.plane_center_scheduler_args = None
        self.plane_sigma_scheduler_args = None
        self.dynamic_gain_scheduler_args = None
        self.dynamic_center_scheduler_args = None
        self.dynamic_sigma_scheduler_args = None

        self.setup_functions()

    def setup_functions(self):
        self.scaling_activation = lambda x: torch.exp(torch.clamp(x, min=-10.0, max=5.0))
        self.scaling_inverse_activation = torch.log

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = lambda x: F.normalize(x, dim=-1)

        # self.gain_mag_activation = F.softplus
        # self.gain_mag_inverse_activation = inverse_softplus

        self.covariance_activation = build_covariance_from_scaling_rotation


    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_plane_center(self):
        return self._plane_center

    @property
    def get_plane_sigma(self):
        return torch.exp(self._plane_log_sigma).clamp(
            min=self.plane_min_sigma,
            max=self.plane_max_sigma,
        )

    # @property
    # def get_gain_mag(self):
    #     return self.gain_mag_activation(self._gain_mag)

    # @property
    # def get_gain_weight(self):
    #     return self.get_opacity * self.get_gain_mag

    def get_covariance(self, scaling_modifier: float = 1.0):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self.get_rotation, return_strip = False
        )

    # ------------------------------------------------------------------
    # Init / save / restore
    # ------------------------------------------------------------------
    def _build_initial_points(self, vertices_path: Optional[str] = None) -> torch.Tensor:
        fused_point_cloud = None

        if vertices_path is not None and os.path.exists(vertices_path):
            try:
                mat = sio.loadmat(vertices_path)
                vertices = mat.get("vertices", None)
                if vertices is not None and vertices.size > 0:
                    base_points = torch.tensor(
                        vertices, dtype = torch.float32, device = self.device
                    )
                    base_count = base_points.shape[0]

                    if base_count > self.target_gaussians:
                        fused_point_cloud = base_points[: self.target_gaussians]
                    else:
                        repeat_idx = torch.randint(
                            0,
                            base_count,
                            (self.target_gaussians,),
                            device = self.device,
                        )
                        jitter = (
                            torch.randn((self.target_gaussians, 3), device = self.device) * 0.01
                        )
                        fused_point_cloud = base_points[repeat_idx] + jitter
            except Exception as exc:
                print(
                    f"Failed to load vertices from {vertices_path}."
                    f"Fallback to random initialization: {exc}"
                )

        if fused_point_cloud is None:
            fused_point_cloud = (
                torch.randn((self.target_gaussians, 3), device = self.device).float() * self.init_range
            )

        return fused_point_cloud
    

    def gaussian_init(self, vertices_path: Optional[str] = None):
        n_points = self.target_gaussians

        beam_center = torch.rand((n_points, 1), device=self.device) * (self.num_beams - 1)
        subc_center = torch.rand((n_points, 1), device=self.device) * (self.num_subcarriers - 1)
        plane_center = torch.cat([beam_center, subc_center], dim=1)

        init_sigma = torch.tensor(
            [self.plane_init_sigma_beam, self.plane_init_sigma_subcarrier],
            device=self.device,
            dtype=torch.float32,
        ).view(1, 2).repeat(n_points, 1)

        opacities_raw = self.inverse_opacity_activation(
            0.1 * torch.ones((n_points, 1), dtype=torch.float32, device=self.device)
        )

        self._plane_center = nn.Parameter(plane_center.requires_grad_(True))
        self._plane_log_sigma = nn.Parameter(torch.log(init_sigma).requires_grad_(True))
        self._opacity = nn.Parameter(opacities_raw.requires_grad_(True))

        self._reset_statistics()
        print(f"[GaussianModel] Number of points at initialization: {n_points}")

    def capture(self):
        return (
            self.target_gaussians,
            self.optimizer_type,
            self.init_range,
            self.num_beams,
            self.num_subcarriers,
            self._plane_center.detach(),
            self._plane_log_sigma.detach(),
            self._opacity.detach(),
            None if self.optimizer is None else self.optimizer.state_dict(),
            self.dynamic_gain_net.state_dict(),
            None if self.dynamic_gain_optimizer is None else self.dynamic_gain_optimizer.state_dict(),
            self.dynamic_center_net.state_dict(),
            self.dynamic_sigma_net.state_dict(),
            None if self.dynamic_center_optimizer is None else self.dynamic_center_optimizer.state_dict(),
            None if self.dynamic_sigma_optimizer is None else self.dynamic_sigma_optimizer.state_dict(),
        )

    def restore(self, model_args, training_args):
        self.target_gaussians = model_args[0]
        self.optimizer_type = model_args[1]
        self.init_range = model_args[2]
        self.num_beams = model_args[3]
        self.num_subcarriers = model_args[4]
        plane_center = model_args[5]
        plane_log_sigma = model_args[6]
        opacity = model_args[7]
        opt_dict = model_args[8] if len(model_args) > 8 else None
        dynamic_gain_net_dict = model_args[9] if len(model_args) > 9 else None
        dynamic_gain_opt_dict = model_args[10] if len(model_args) > 10 else None
        dynamic_center_net_dict = model_args[11] if len(model_args) > 11 else None
        dynamic_sigma_net_dict = model_args[12] if len(model_args) > 12 else None
        dynamic_center_opt_dict = model_args[13] if len(model_args) > 13 else None
        dynamic_sigma_opt_dict = model_args[14] if len(model_args) > 14 else None

        self._plane_center = nn.Parameter(plane_center.to(self.device).requires_grad_(True))
        self._plane_log_sigma = nn.Parameter(plane_log_sigma.to(self.device).requires_grad_(True))
        self._opacity = nn.Parameter(opacity.to(self.device).requires_grad_(True))

        self._reset_statistics()
        self.training_setup(training_args)

        if opt_dict is not None and self.optimizer is not None:
            self.optimizer.load_state_dict(opt_dict)

        if dynamic_gain_net_dict is not None:
            self.dynamic_gain_net.load_state_dict(dynamic_gain_net_dict)

        if dynamic_center_net_dict is not None:
            self.dynamic_center_net.load_state_dict(dynamic_center_net_dict)

        if dynamic_sigma_net_dict is not None:
            self.dynamic_sigma_net.load_state_dict(dynamic_sigma_net_dict)

        if dynamic_gain_opt_dict is not None and self.dynamic_gain_optimizer is not None:
            self.dynamic_gain_optimizer.load_state_dict(dynamic_gain_opt_dict)

        if dynamic_center_opt_dict is not None and self.dynamic_center_optimizer is not None:
            self.dynamic_center_optimizer.load_state_dict(dynamic_center_opt_dict)

        if dynamic_sigma_opt_dict is not None and self.dynamic_sigma_optimizer is not None:
            self.dynamic_sigma_optimizer.load_state_dict(dynamic_sigma_opt_dict)

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    def _reset_statistics(self):
        n = self._plane_center.shape[0]
        self.importance_accum = torch.zeros((n, 1), device=self.device)
        self.importance_denom = torch.zeros((n, 1), device=self.device)

    def training_setup(self, training_args):
        self._reset_statistics()

        param_groups = [
            {"params": [self._plane_center], "lr": training_args.plane_center_lr, "name": "plane_center"},
            {"params": [self._plane_log_sigma], "lr": training_args.plane_sigma_lr, "name": "plane_sigma"},
            {"params": [self._opacity], "lr": training_args.opacity_lr, "name": "opacity"},
        ]

        if getattr(training_args, "optimizer_type", self.optimizer_type) == "adamw":
            self.optimizer = torch.optim.AdamW(param_groups, lr=0.0, eps=1e-8)
        else:
            self.optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-8)

        self.plane_center_scheduler_args = get_expon_lr_func(
            lr_init=training_args.plane_center_lr,
            lr_final=getattr(training_args, "plane_center_lr_final", training_args.plane_center_lr),
            lr_delay_mult=1.0,
            max_steps=training_args.iterations,
        )
        self.plane_sigma_scheduler_args = get_expon_lr_func(
            lr_init=training_args.plane_sigma_lr,
            lr_final=getattr(training_args, "plane_sigma_lr_final", training_args.plane_sigma_lr),
            lr_delay_mult=1.0,
            max_steps=training_args.iterations,
        )

        self.dynamic_gain_optimizer = torch.optim.Adam(
            self.dynamic_gain_net.parameters(),
            lr=getattr(training_args, "dynamic_gain_lr", 0.001),
            eps=1e-8,
        )
        self.dynamic_center_optimizer = torch.optim.Adam(
            self.dynamic_center_net.parameters(),
            lr=getattr(training_args, "dynamic_center_lr", 0.001),
            eps=1e-8,
        )
        self.dynamic_sigma_optimizer = torch.optim.Adam(
            self.dynamic_sigma_net.parameters(),
            lr=getattr(training_args, "dynamic_sigma_lr", 0.001),
            eps=1e-8,
        )

        self.dynamic_gain_scheduler_args = get_expon_lr_func(
            lr_init=getattr(training_args, "dynamic_gain_lr", 0.001),
            lr_final=getattr(training_args, "dynamic_gain_lr_final", 0.0001),
            lr_delay_mult=1.0,
            max_steps=training_args.iterations,
        )
        self.dynamic_center_scheduler_args = get_expon_lr_func(
            lr_init=getattr(training_args, "dynamic_center_lr", 0.001),
            lr_final=getattr(training_args, "dynamic_center_lr_final", 0.0001),
            lr_delay_mult=1.0,
            max_steps=training_args.iterations,
        )
        self.dynamic_sigma_scheduler_args = get_expon_lr_func(
            lr_init=getattr(training_args, "dynamic_sigma_lr", 0.001),
            lr_final=getattr(training_args, "dynamic_sigma_lr_final", 0.0001),
            lr_delay_mult=1.0,
            max_steps=training_args.iterations,
        )

    def update_learning_rate(self, iteration):
        if self.optimizer is not None:
            plane_center_lr = self.plane_center_scheduler_args(iteration)
            plane_sigma_lr = self.plane_sigma_scheduler_args(iteration)
            for param_group in self.optimizer.param_groups:
                if param_group.get("name") == "plane_center":
                    param_group["lr"] = plane_center_lr
                elif param_group.get("name") == "plane_sigma":
                    param_group["lr"] = plane_sigma_lr

        if self.dynamic_gain_optimizer is not None:
            gain_lr = self.dynamic_gain_scheduler_args(iteration)
            for param_group in self.dynamic_gain_optimizer.param_groups:
                param_group["lr"] = gain_lr
        if self.dynamic_center_optimizer is not None:
            center_lr = self.dynamic_center_scheduler_args(iteration)
            for param_group in self.dynamic_center_optimizer.param_groups:
                param_group["lr"] = center_lr
        if self.dynamic_sigma_optimizer is not None:
            sigma_lr = self.dynamic_sigma_scheduler_args(iteration)
            for param_group in self.dynamic_sigma_optimizer.param_groups:
                param_group["lr"] = sigma_lr

    def _build_condition_feature(self, rx_pos: torch.Tensor) -> torch.Tensor:
        rx = rx_pos.view(1, 3).to(self.device, dtype=self.get_xyz.dtype)

        xyz = self.get_xyz
        rel = xyz - rx
        dist = torch.norm(rel, dim=-1, keepdim=True).clamp(min=1e-6)
        rx_rep = rx.expand(xyz.shape[0], -1)

        feat = torch.cat(
            [xyz, rx_rep, rel, torch.log1p(dist)],
            dim=-1,
        )
        return feat

    def get_dynamic_gain_weight(self, rx_pos: torch.Tensor) -> torch.Tensor:
        rx_pos = rx_pos.view(1, 3).to(self.device, dtype=self._plane_center.dtype)
        rx_expand = rx_pos.expand(self._plane_center.shape[0], -1)

        beam_norm, subc_norm = self._get_normalized_plane_center()
        plane_coord_norm = torch.cat([beam_norm, subc_norm], dim=-1)

        dyn = F.softplus(self.dynamic_gain_net(plane_coord_norm, rx_expand))
        gain = self.get_opacity * dyn
        return gain

    def _get_normalized_plane_center(self):
        beam_norm = 2.0 * self._plane_center[:, 0:1] / max(self.num_beams - 1, 1) - 1.0
        subc_norm = 2.0 * self._plane_center[:, 1:2] / max(self.num_subcarriers - 1, 1) - 1.0
        return beam_norm, subc_norm

    def get_dynamic_plane_center(self, rx_pos: torch.Tensor) -> torch.Tensor:
        if not self.use_dynamic_center:
            return self.get_plane_center

        rx_pos = rx_pos.view(1, 3).to(self.device, dtype=self._plane_center.dtype)
        rx_expand = rx_pos.expand(self._plane_center.shape[0], -1)

        beam_norm, subc_norm = self._get_normalized_plane_center()
        plane_coord_norm = torch.cat([beam_norm, subc_norm], dim=-1)

        raw_delta_center = self.dynamic_center_net(plane_coord_norm, rx_expand)
        bounded_delta = torch.tanh(raw_delta_center)

        delta_beam = bounded_delta[:, 0] * self.center_shift_max_beam
        delta_subc = bounded_delta[:, 1] * self.center_shift_max_subcarrier
        delta_center = torch.stack([delta_beam, delta_subc], dim=-1)

        dynamic_center = self._plane_center + delta_center
        beam_center = dynamic_center[:, 0].clamp(
            min=0.0, max=float(max(self.num_beams - 1, 0))
        )
        subc_center = dynamic_center[:, 1].clamp(
            min=0.0, max=float(max(self.num_subcarriers - 1, 0))
        )
        return torch.stack([beam_center, subc_center], dim=-1)

    def get_dynamic_plane_sigma(self, rx_pos: torch.Tensor) -> torch.Tensor:
        if not self.use_dynamic_sigma:
            return self.get_plane_sigma

        rx_pos = rx_pos.view(1, 3).to(self.device, dtype=self._plane_center.dtype)
        rx_expand = rx_pos.expand(self._plane_center.shape[0], -1)

        beam_norm, subc_norm = self._get_normalized_plane_center()
        plane_coord_norm = torch.cat([beam_norm, subc_norm], dim=-1)

        raw_delta_log_sigma = self.dynamic_sigma_net(plane_coord_norm, rx_expand)
        bounded_delta = torch.tanh(raw_delta_log_sigma)

        delta_log_sigma_beam = bounded_delta[:, 0] * self.sigma_log_shift_max_beam
        delta_log_sigma_subc = bounded_delta[:, 1] * self.sigma_log_shift_max_subcarrier
        delta_log_sigma = torch.stack([delta_log_sigma_beam, delta_log_sigma_subc], dim=-1)

        dynamic_log_sigma = self._plane_log_sigma + delta_log_sigma
        dynamic_sigma = torch.exp(dynamic_log_sigma).clamp(
            min=self.plane_min_sigma, max=self.plane_max_sigma
        )
        return dynamic_sigma

    def get_dynamic_spectral_profile(self, rx_pos, num_subcarriers=None):
        feat = self._build_condition_feature(rx_pos)
        coeff = self.dynamic_spectral_net(feat)

        K = self.num_subcarriers if num_subcarriers is None else int(num_subcarriers)
        basis = self.spectral_basis[:, :K]

        spectral_logits = coeff @ basis
        spectral_profile = F.softplus(spectral_logits)

        spectral_profile = spectral_profile / spectral_profile.mean(
            dim=-1, keepdim=True
        ).clamp(min=1e-6)

        return spectral_profile

    # ------------------------------------------------------------------
    # Statistics for pruning / densification
    # ------------------------------------------------------------------

    def accumulate_training_stats(self, importance=None):
        if importance is None:
            return
        imp = importance.detach().reshape(-1, 1)
        self.importance_accum += imp
        self.importance_denom += 1.0

    def get_avg_xyz_grad(self):
        denom = torch.clamp(self.grad_denom, min=1.0)
        return self.xyz_gradient_accum / denom

    def get_avg_importance(self):
        denom = torch.clamp(self.importance_denom, min=1.0)
        return self.importance_accum / denom

    # ------------------------------------------------------------------
    # PLY I/O
    # ------------------------------------------------------------------
    def construct_list_of_attributes(self):
        attrs = ["x", "y", "z", "nx", "ny", "nz", "opacity"]
        attrs += [f"scale_{i}" for i in range(3)]
        attrs += [f"rot_{i}" for i in range(4)]
        # attrs += ["gain_mag"]
        return attrs

    def save_ply(self, path):


        os.makedirs(os.path.dirname(path), exist_ok=True)

        centers = self._plane_center.detach().cpu().numpy()          # (N,2)
        sigmas = self.get_plane_sigma.detach().cpu().numpy()         # (N,2)
        opacity = self.get_opacity.detach().cpu().numpy().reshape(-1)

        n = centers.shape[0]

        # pseudo-3D coordinates for visualization / compatibility
        # x = beam center, y = subcarrier center, z = 0
        xyz = np.zeros((n, 3), dtype=np.float32)
        xyz[:, 0] = centers[:, 0]
        xyz[:, 1] = centers[:, 1]
        xyz[:, 2] = 0.0

        # store sigma info too
        sigma_beam = sigmas[:, 0].astype(np.float32)
        sigma_subc = sigmas[:, 1].astype(np.float32)
        opacity = opacity.astype(np.float32)

        vertex_data = np.empty(
            n,
            dtype=[
                ("x", "f4"),
                ("y", "f4"),
                ("z", "f4"),
                ("opacity", "f4"),
                ("sigma_beam", "f4"),
                ("sigma_subcarrier", "f4"),
            ],
        )

        vertex_data["x"] = xyz[:, 0]
        vertex_data["y"] = xyz[:, 1]
        vertex_data["z"] = xyz[:, 2]
        vertex_data["opacity"] = opacity
        vertex_data["sigma_beam"] = sigma_beam
        vertex_data["sigma_subcarrier"] = sigma_subc

        ply = PlyData([PlyElement.describe(vertex_data, "vertex")], text=True)
        ply.write(path)

    def load_ply(self, path: str):
        plydata = PlyData.read(path)

        xyz = np.stack(
            [
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ],
            axis = 1,
        )

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)), dtype=np.float32)
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot_")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)), dtype=np.float32)
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # gain_mag = np.asarray(plydata.elements[0]["gain_mag"])[..., np.newaxis]

        xyz_t = torch.tensor(xyz, dtype=torch.float32, device=self.device)
        opacity_t = torch.tensor(opacities, dtype=torch.float32, device=self.device)
        scale_t = torch.tensor(scales, dtype=torch.float32, device=self.device)
        rot_t = torch.tensor(rots, dtype=torch.float32, device=self.device)
        # gain_mag_t = torch.tensor(gain_mag, dtype=torch.float32, device=self.device)

        self._xyz = nn.Parameter(xyz_t.requires_grad_(True))
        self._opacity = nn.Parameter(
            self.inverse_opacity_activation(opacity_t).requires_grad_(True)
        )
        self._scaling = nn.Parameter(
            self.scaling_inverse_activation(torch.clamp(scale_t, min=1e-8)).requires_grad_(True)
        )
        self._rotation = nn.Parameter(rot_t.requires_grad_(True))
        # self._gain_mag = nn.Parameter(
        #     self.gain_mag_inverse_activation(torch.clamp(gain_mag_t, min=1e-8)).requires_grad_(True)
        # )

        self._reset_statistics()

    # ------------------------------------------------------------------
    # Optimizer-safe tensor replacement helpers
    # ------------------------------------------------------------------

    def replace_tensor_to_optimizer(self, tensor: torch.Tensor, name:str):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] != name:
                continue

            stored_state = self.optimizer.state.get(group["params"][0], None)

            if stored_state is not None:
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

            else:
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))

            optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask: torch.Tensor):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            old_param = group["params"][0]
            stored_state = self.optimizer.state.get(old_param, None)

            new_tensor = old_param[mask]

            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[old_param]
                group["params"][0] = nn.Parameter(new_tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state
            else:
                group["params"][0] = nn.Parameter(new_tensor.requires_grad_(True))

            optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def cat_tensors_to_optimizer(self, tensors_dict: Dict[str, torch.Tensor]):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            extension_tensor = tensors_dict[group["name"]]
            old_param = group["params"][0]
            stored_state = self.optimizer.state.get(old_param, None)

            new_tensor = torch.cat((old_param, extension_tensor), dim=0)

            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0
                )

                del self.optimizer.state[old_param]
                group["params"][0] = nn.Parameter(new_tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state
            else:
                group["params"][0] = nn.Parameter(new_tensor.requires_grad_(True))

            optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    # ------------------------------------------------------------------
    # Prune / densify
    # ------------------------------------------------------------------

    def prune_points(self, mask: torch.Tensor):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        # self._gain_mag = optimizable_tensors["gain_mag"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.grad_denom = self.grad_denom[valid_points_mask]
        self.importance_accum = self.importance_accum[valid_points_mask]
        self.importance_denom = self.importance_denom[valid_points_mask]

    def densification_postfix(
        self,
        new_xyz: torch.Tensor,
        new_opacity: torch.Tensor,
        new_scaling: torch.Tensor,
        new_rotation: torch.Tensor,
        # new_gain_mag: torch.Tensor,
    ):
        tensors_dict = {
            "xyz": new_xyz,
            "opacity": new_opacity,
            "scaling": new_scaling,
            "rotation": new_rotation,
            # "gain_mag": new_gain_mag,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(tensors_dict)

        self._xyz = optimizable_tensors["xyz"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        # self._gain_mag = optimizable_tensors["gain_mag"]

        self._reset_statistics()

    def densify_and_clone(
        self,
        grads: torch.Tensor,
        grad_threshold: float,
        clone_scale_threshold: float,
        importance_threshold: float = 0.0,
    ):
        avg_importance = self.get_avg_importance().squeeze(-1)
        selected_pts_mask = (grads.squeeze(-1) >= grad_threshold)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            self.get_scaling.max(dim=1).values <= clone_scale_threshold,
        )

        if importance_threshold > 0.0:
            selected_pts_mask = torch.logical_and(
                selected_pts_mask, avg_importance >= importance_threshold
            )

        if selected_pts_mask.sum() == 0:
            return

        new_xyz = self._xyz[selected_pts_mask].clone()
        new_opacity = self._opacity[selected_pts_mask].clone()
        new_scaling = self._scaling[selected_pts_mask].clone()
        new_rotation = self._rotation[selected_pts_mask].clone()
        # new_gain_mag = self._gain_mag[selected_pts_mask].clone()


        self.densification_postfix(
            new_xyz, new_opacity, new_scaling, new_rotation
        )

    def densify_and_split(
        self,
        grads: torch.Tensor,
        grad_threshold: float,
        split_scale_threshold: float,
        importance_threshold: float = 0.0,
        n_splits: int = 2,
    ):
        avg_importance = self.get_avg_importance().squeeze(-1)
        selected_pts_mask = (grads.squeeze(-1) >= grad_threshold)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            self.get_scaling.max(dim=1).values > split_scale_threshold,
        )
        if importance_threshold > 0.0:
            selected_pts_mask = torch.logical_and(
                selected_pts_mask, avg_importance >= importance_threshold
            )

        n_selected = int(selected_pts_mask.sum().item())
        if n_selected == 0:
            return

        stds = self.get_scaling[selected_pts_mask].repeat(n_splits, 1)
        means = torch.zeros((stds.size(0), 3), device=self.device)
        samples = torch.normal(mean=means, std=stds)

        rots = self.get_rotation[selected_pts_mask].repeat(n_splits,1)
        from utils.general_utils import build_rotation  # local import to avoid clutter
        rot_mats = build_rotation(rots)

        base_xyz = self.get_xyz[selected_pts_mask].repeat(n_splits, 1)
        new_xyz = torch.bmm(rot_mats, samples.unsqueeze(-1)).squeeze(-1) + base_xyz

        new_scaling = self.scaling_inverse_activation(
            torch.clamp(
                self.get_scaling[selected_pts_mask].repeat(n_splits, 1) / (0.8 * n_splits),
                min=1e-8,
            )
        )
        new_rotation = self.get_rotation[selected_pts_mask].repeat(n_splits, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(n_splits, 1)
        # new_gain_mag = self._gain_mag[selected_pts_mask].repeat(n_splits, 1)

        self.densification_postfix(
            new_xyz, new_opacity, new_scaling, new_rotation
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(n_splits * n_selected, device = self.device, dtype = torch.bool)
            )
        )
        self.prune_points(prune_filter)

    def densify_and_prune(
        self,
        max_grad: float,
        min_opacity: float,
        # min_gain_mag: float,
        clone_scale_threshold: float,
        split_scale_threshold: float,
        importance_threshold: float = 0.0,
        max_scale: Optional[float] = None,
        n_splits: int = 2,
    ):
        grads = self.get_avg_xyz_grad()
        grads[torch.isnan(grads)] = 0.0

        self.densify_and_clone(
            grads=grads,
            grad_threshold = max_grad,
            clone_scale_threshold = clone_scale_threshold,
            importance_threshold = importance_threshold,
        )

        self.densify_and_split(
            grads = grads,
            grad_threshold = max_grad,
            split_scale_threshold = split_scale_threshold,
            importance_threshold = importance_threshold,
            n_splits = n_splits,
        )

        prune_mask = (self.get_opacity.squeeze(-1) < min_opacity)
        # prune_mask = torch.logical_or(
        #     prune_mask, self.get_gain_mag.squeeze(-1) < min_gain_mag
        # )

        if max_scale is not None:
            prune_mask = torch.logical_or(
                prune_mask, self.get_scaling.max(dim=1).values > max_scale
            )

        self.prune_points(prune_mask)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def reset_opacity(self, max_opacity: float = 0.01):
        new_opacity = self.inverse_opacity_activation(
            torch.minimum(
                self.get_opacity,
                torch.full_like(self.get_opacity, max_opacity)
            )
        )
        optimizable = self.replace_tensor_to_optimizer(new_opacity, "opacity")
        self._opacity = optimizable["opacity"]