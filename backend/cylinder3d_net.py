# -*- coding: utf-8 -*-
"""
cylinder3d_net.py
-----------------
Self-contained Cylinder3D model definition and cylindrical voxelisation
pre-processing.  Architecture is an exact structural match to the official
checkpoint at weights/model_load.pth

References
----------
  Cylinder3D — Zhu et al., CVPR 2021 (Oral)
  https://github.com/xinge008/Cylinder3D
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# spconv 2.x (spconv-cu121) is installed — always import from spconv.pytorch
import spconv.pytorch as spconv  # type: ignore[import]

try:
    import torch_scatter  # type: ignore[import]
    _HAS_SCATTER = True
except ImportError:
    _HAS_SCATTER = False

# ---------------------------------------------------------------------------
# Cylindrical Voxelisation Config
# ---------------------------------------------------------------------------
GRID_SIZE       = np.array([480, 360, 32], dtype=np.int32)
MAX_BOUND       = np.array([50.0,  np.pi,  2.0], dtype=np.float32)
MIN_BOUND       = np.array([0.0,  -np.pi, -4.0], dtype=np.float32)
MAX_PT_PER_VOX  = 64
FEA_DIM         = 9       # per-point feature dimensionality
OUT_PT_FEA_DIM  = 256
FEA_COMPRE      = 16      # compressed voxel feature size fed to spconv
N_CLASSES       = 20
INIT_SIZE       = 32


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------
def cart2polar(xyz: np.ndarray) -> np.ndarray:
    """Cartesian (x, y, z) → cylindrical (ρ, φ, z)."""
    rho = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2)
    phi = np.arctan2(xyz[:, 1], xyz[:, 0])
    return np.stack([rho, phi, xyz[:, 2]], axis=1)


# ---------------------------------------------------------------------------
# CylinderVoxelizer
# ---------------------------------------------------------------------------
class CylinderVoxelizer:
    """
    Convert a raw point cloud (N, 4) [x, y, z, intensity] to cylindrical
    voxel indices and per-point feature vectors ready for cylinder_fea.
    """

    def __init__(
        self,
        grid_size: np.ndarray = GRID_SIZE,
        max_bound: np.ndarray = MAX_BOUND,
        min_bound: np.ndarray = MIN_BOUND,
    ):
        self.grid_size  = grid_size
        self.max_bound  = max_bound
        self.min_bound  = min_bound
        self.intervals  = (max_bound - min_bound) / (grid_size - 1)

    def __call__(self, points: np.ndarray):
        """
        Parameters
        ----------
        points : np.ndarray, shape (N, 4)  [x, y, z, intensity]

        Returns
        -------
        pt_fea   : torch.FloatTensor  (N, 9)
        grid_ind : torch.IntTensor    (N, 3)
        """
        xyz = points[:, :3].astype(np.float32)
        sig = points[:, 3].astype(np.float32)

        # Cylindrical coordinates
        xyz_pol = cart2polar(xyz)

        # Clip to valid volume
        xyz_pol_clipped = np.clip(xyz_pol, self.min_bound, self.max_bound)

        # Voxel indices (integer)
        grid_ind = np.floor(
            (xyz_pol_clipped - self.min_bound) / self.intervals
        ).astype(np.int32)
        grid_ind = np.clip(grid_ind, 0, self.grid_size - 1)

        # Voxel centre in polar space (for offset computation)
        voxel_centers = (grid_ind.astype(np.float32) + 0.5) * self.intervals + self.min_bound

        # 9-D per-point feature (matches training pipeline):
        # [Δρ, Δφ, Δz,  ρ,  φ,  z,  x_cart,  y_cart,  intensity]
        return_xyz   = xyz_pol - voxel_centers                # (N, 3) offsets
        return_xyz   = np.concatenate([return_xyz, xyz_pol, xyz[:, :2]], axis=1)  # (N, 8)
        return_fea   = np.concatenate([return_xyz, sig[:, None]], axis=1)         # (N, 9)

        pt_fea_tensor   = torch.from_numpy(return_fea).float()
        grid_ind_tensor = torch.from_numpy(grid_ind).int()
        return pt_fea_tensor, grid_ind_tensor


# ---------------------------------------------------------------------------
# cylinder_fea  (PointNet-style voxel encoder)
# ---------------------------------------------------------------------------
class cylinder_fea(nn.Module):
    """Voxel-level feature aggregation using scatter-max."""

    def __init__(
        self,
        grid_size,
        fea_dim: int = FEA_DIM,
        out_pt_fea_dim: int = OUT_PT_FEA_DIM,
        max_pt_per_encode: int = MAX_PT_PER_VOX,
        fea_compre: int = FEA_COMPRE,
    ):
        super().__init__()
        self.PPmodel = nn.Sequential(
            nn.BatchNorm1d(fea_dim),
            nn.Linear(fea_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, out_pt_fea_dim),
        )
        self.max_pt     = max_pt_per_encode
        self.fea_compre = fea_compre
        self.grid_size  = grid_size
        self.pool_dim   = out_pt_fea_dim

        self.local_pool_op = nn.MaxPool2d(3, stride=1, padding=1, dilation=1)

        if fea_compre is not None:
            self.fea_compression = nn.Sequential(
                nn.Linear(self.pool_dim, fea_compre),
                nn.ReLU(),
            )
            self.pt_fea_dim = fea_compre
        else:
            self.pt_fea_dim = self.pool_dim

    def forward(self, pt_fea: list, xy_ind: list):
        dev = pt_fea[0].device

        # Batch-prepend index
        cat_pt_ind = [
            F.pad(xy_ind[b], (1, 0), "constant", value=b)
            for b in range(len(xy_ind))
        ]
        cat_pt_fea = torch.cat(pt_fea, dim=0)
        cat_pt_ind = torch.cat(cat_pt_ind, dim=0)
        pt_num     = cat_pt_ind.shape[0]

        # Shuffle
        shuffled   = torch.randperm(pt_num, device=dev)
        cat_pt_fea = cat_pt_fea[shuffled]
        cat_pt_ind = cat_pt_ind[shuffled]

        # Unique voxel indices
        unq, unq_inv, _ = torch.unique(cat_pt_ind, return_inverse=True,
                                        return_counts=True, dim=0)
        unq = unq.long()

        # MLP
        proc_fea = self.PPmodel(cat_pt_fea)

        # Scatter-max aggregation
        if _HAS_SCATTER:
            pooled = torch_scatter.scatter_max(proc_fea, unq_inv, dim=0)[0]
        else:
            # CPU fallback (slow but functional)
            pooled = torch.zeros(unq.shape[0], proc_fea.shape[1], device=dev)
            pooled.scatter_reduce_(0, unq_inv.unsqueeze(1).expand_as(proc_fea),
                                   proc_fea, reduce="amax", include_self=True)

        if self.fea_compre is not None:
            pooled = self.fea_compression(pooled)

        return unq, pooled


# ---------------------------------------------------------------------------
# Asymmetric spconv blocks
# ---------------------------------------------------------------------------
def _conv3x3(i, o, stride=1, key=None):
    k = f"{key}_33" if key else None
    return spconv.SubMConv3d(i, o, 3, stride=stride, padding=1, bias=False, indice_key=k)

def _conv1x3(i, o, stride=1, key=None):
    k = f"{key}_13" if key else None
    return spconv.SubMConv3d(i, o, (1, 3, 3), stride=stride, padding=(0, 1, 1), bias=False, indice_key=k)

def _conv3x1(i, o, stride=1, key=None):
    k = f"{key}_31" if key else None
    return spconv.SubMConv3d(i, o, (3, 1, 3), stride=stride, padding=(1, 0, 1), bias=False, indice_key=k)

def _conv3x1x1(i, o, stride=1, key=None):
    k = f"{key}_311" if key else None
    return spconv.SubMConv3d(i, o, (3, 1, 1), stride=stride, padding=(1, 0, 0), bias=False, indice_key=k)

def _conv1x3x1(i, o, stride=1, key=None):
    k = f"{key}_131" if key else None
    return spconv.SubMConv3d(i, o, (1, 3, 1), stride=stride, padding=(0, 1, 0), bias=False, indice_key=k)

def _conv1x1x3(i, o, stride=1, key=None):
    k = f"{key}_113" if key else None
    return spconv.SubMConv3d(i, o, (1, 1, 3), stride=stride, padding=(0, 0, 1), bias=False, indice_key=k)


class ResContextBlock(nn.Module):
    def __init__(self, in_f, out_f, indice_key=None):
        super().__init__()
        k = indice_key
        self.conv1   = _conv1x3(in_f,  out_f, key=k + "bef")
        self.bn0     = nn.BatchNorm1d(out_f);  self.act1   = nn.LeakyReLU()
        self.conv1_2 = _conv3x1(out_f, out_f, key=k + "bef")
        self.bn0_2   = nn.BatchNorm1d(out_f);  self.act1_2 = nn.LeakyReLU()
        self.conv2   = _conv3x1(in_f,  out_f, key=k + "bef")
        self.bn1     = nn.BatchNorm1d(out_f);  self.act2   = nn.LeakyReLU()
        self.conv3   = _conv1x3(out_f, out_f, key=k + "bef")
        self.bn2     = nn.BatchNorm1d(out_f);  self.act3   = nn.LeakyReLU()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)

    def forward(self, x):
        s = self.conv1(x);   s = s.replace_feature(self.act1(self.bn0(s.features)))
        s = self.conv1_2(s); s = s.replace_feature(self.act1_2(self.bn0_2(s.features)))
        r = self.conv2(x);   r = r.replace_feature(self.act2(self.bn1(r.features)))
        r = self.conv3(r);   r = r.replace_feature(self.act3(self.bn2(r.features)))
        r = r.replace_feature(r.features + s.features)
        return r


class ResBlock(nn.Module):
    def __init__(self, in_f, out_f, dropout_rate=0.2,
                 pooling=True, height_pooling=False, indice_key=None):
        super().__init__()
        self.pooling = pooling
        k = indice_key
        self.conv1   = _conv3x1(in_f,  out_f, key=k + "bef"); self.act1   = nn.LeakyReLU(); self.bn0   = nn.BatchNorm1d(out_f)
        self.conv1_2 = _conv1x3(out_f, out_f, key=k + "bef"); self.act1_2 = nn.LeakyReLU(); self.bn0_2 = nn.BatchNorm1d(out_f)
        self.conv2   = _conv1x3(in_f,  out_f, key=k + "bef"); self.act2   = nn.LeakyReLU(); self.bn1   = nn.BatchNorm1d(out_f)
        self.conv3   = _conv3x1(out_f, out_f, key=k + "bef"); self.act3   = nn.LeakyReLU(); self.bn2   = nn.BatchNorm1d(out_f)
        if pooling:
            stride = (2, 2, 1) if not height_pooling else 2
            self.pool = spconv.SparseConv3d(out_f, out_f, 3, stride=stride,
                                            padding=1, indice_key=k, bias=False)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)

    def forward(self, x):
        s = self.conv1(x);   s = s.replace_feature(self.act1(self.bn0(s.features)))
        s = self.conv1_2(s); s = s.replace_feature(self.act1_2(self.bn0_2(s.features)))
        r = self.conv2(x);   r = r.replace_feature(self.act2(self.bn1(r.features)))
        r = self.conv3(r);   r = r.replace_feature(self.act3(self.bn2(r.features)))
        r = r.replace_feature(r.features + s.features)
        if self.pooling:
            return self.pool(r), r
        return r


class UpBlock(nn.Module):
    def __init__(self, in_f, out_f, indice_key=None, up_key=None):
        super().__init__()
        k = indice_key
        self.trans_dilao = _conv3x3(in_f,  out_f, key=k + "new_up"); self.trans_act = nn.LeakyReLU(); self.trans_bn = nn.BatchNorm1d(out_f)
        self.conv1       = _conv1x3(out_f, out_f, key=k);             self.act1      = nn.LeakyReLU(); self.bn1      = nn.BatchNorm1d(out_f)
        self.conv2       = _conv3x1(out_f, out_f, key=k);             self.act2      = nn.LeakyReLU(); self.bn2      = nn.BatchNorm1d(out_f)
        self.conv3       = _conv3x3(out_f, out_f, key=k);             self.act3      = nn.LeakyReLU(); self.bn3      = nn.BatchNorm1d(out_f)
        self.up_subm     = spconv.SparseInverseConv3d(out_f, out_f, 3, indice_key=up_key, bias=False)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)

    def forward(self, x, skip):
        a = self.trans_dilao(x); a = a.replace_feature(self.trans_act(self.trans_bn(a.features)))
        a = self.up_subm(a);     a = a.replace_feature(a.features + skip.features)
        a = self.conv1(a);       a = a.replace_feature(self.act1(self.bn1(a.features)))
        a = self.conv2(a);       a = a.replace_feature(self.act2(self.bn2(a.features)))
        a = self.conv3(a);       a = a.replace_feature(self.act3(self.bn3(a.features)))
        return a


class ReconBlock(nn.Module):
    def __init__(self, in_f, out_f, indice_key=None):
        super().__init__()
        k = indice_key
        self.conv1   = _conv3x1x1(in_f, out_f, key=k + "bef"); self.bn0   = nn.BatchNorm1d(out_f); self.act1   = nn.Sigmoid()
        self.conv1_2 = _conv1x3x1(in_f, out_f, key=k + "bef"); self.bn0_2 = nn.BatchNorm1d(out_f); self.act1_2 = nn.Sigmoid()
        self.conv1_3 = _conv1x1x3(in_f, out_f, key=k + "bef"); self.bn0_3 = nn.BatchNorm1d(out_f); self.act1_3 = nn.Sigmoid()

    def forward(self, x):
        s1 = self.conv1(x);   s1 = s1.replace_feature(self.act1(self.bn0(s1.features)))
        s2 = self.conv1_2(x); s2 = s2.replace_feature(self.act1_2(self.bn0_2(s2.features)))
        s3 = self.conv1_3(x); s3 = s3.replace_feature(self.act1_3(self.bn0_3(s3.features)))
        s1 = s1.replace_feature((s1.features + s2.features + s3.features) * x.features)
        return s1


# ---------------------------------------------------------------------------
# Asymm_3d_spconv  (encoder-decoder segmentation head)
# ---------------------------------------------------------------------------
class Asymm_3d_spconv(nn.Module):
    def __init__(self, output_shape, num_input_features=FEA_COMPRE,
                 nclasses=N_CLASSES, n_height=32, init_size=INIT_SIZE):
        super().__init__()
        self.nclasses    = nclasses
        self.sparse_shape = np.array(output_shape)

        s = init_size
        self.downCntx  = ResContextBlock(num_input_features, s,    indice_key="pre")
        self.resBlock2 = ResBlock(s,    2*s,  height_pooling=True,  indice_key="down2")
        self.resBlock3 = ResBlock(2*s,  4*s,  height_pooling=True,  indice_key="down3")
        self.resBlock4 = ResBlock(4*s,  8*s,  height_pooling=False, indice_key="down4")
        self.resBlock5 = ResBlock(8*s,  16*s, height_pooling=False, indice_key="down5")

        self.upBlock0  = UpBlock(16*s, 16*s, indice_key="up0", up_key="down5")
        self.upBlock1  = UpBlock(16*s, 8*s,  indice_key="up1", up_key="down4")
        self.upBlock2  = UpBlock(8*s,  4*s,  indice_key="up2", up_key="down3")
        self.upBlock3  = UpBlock(4*s,  2*s,  indice_key="up3", up_key="down2")

        self.ReconNet  = ReconBlock(2*s, 2*s, indice_key="recon")

        self.logits    = spconv.SubMConv3d(4*s, nclasses,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=True, indice_key="logit")

    def forward(self, voxel_features, coors, batch_size):
        coors = coors.int()
        ret   = spconv.SparseConvTensor(voxel_features, coors,
                                        self.sparse_shape, batch_size)
        ret              = self.downCntx(ret)
        down1c, down1b   = self.resBlock2(ret)
        down2c, down2b   = self.resBlock3(down1c)
        down3c, down3b   = self.resBlock4(down2c)
        down4c, down4b   = self.resBlock5(down3c)
        up4e             = self.upBlock0(down4c, down4b)
        up3e             = self.upBlock1(up4e,   down3b)
        up2e             = self.upBlock2(up3e,   down2b)
        up1e             = self.upBlock3(up2e,   down1b)
        up0e             = self.ReconNet(up1e)
        up0e             = up0e.replace_feature(torch.cat([up0e.features, up1e.features], dim=1))
        logits           = self.logits(up0e)
        return logits.dense()          # (B, nclasses, 480, 360, 32)


# ---------------------------------------------------------------------------
# cylinder_asym  (top-level container)
# ---------------------------------------------------------------------------
class cylinder_asym(nn.Module):
    def __init__(self, cylin_model, segmentator_spconv, sparse_shape):
        super().__init__()
        self.cylinder_3d_generator   = cylin_model
        self.cylinder_3d_spconv_seg  = segmentator_spconv
        self.sparse_shape            = sparse_shape

    def forward(self, pt_fea: list, vox_ind: list, batch_size: int):
        coords, features_3d = self.cylinder_3d_generator(pt_fea, vox_ind)
        spatial_features    = self.cylinder_3d_spconv_seg(features_3d, coords, batch_size)
        return spatial_features


# ---------------------------------------------------------------------------
# build_model
# ---------------------------------------------------------------------------
def build_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    """
    Instantiate the Cylinder3D model with default SemanticKITTI hyperparameters,
    load the pretrained checkpoint, and return in eval mode.
    """
    grid_size   = GRID_SIZE.tolist()     # [480, 360, 32]

    cylin_model = cylinder_fea(
        grid_size       = grid_size,
        fea_dim         = FEA_DIM,
        out_pt_fea_dim  = OUT_PT_FEA_DIM,
        max_pt_per_encode = MAX_PT_PER_VOX,
        fea_compre      = FEA_COMPRE,
    )

    segmentator = Asymm_3d_spconv(
        output_shape        = grid_size,
        num_input_features  = FEA_COMPRE,
        nclasses            = N_CLASSES,
        n_height            = GRID_SIZE[2],
        init_size           = INIT_SIZE,
    )

    model = cylinder_asym(
        cylin_model         = cylin_model,
        segmentator_spconv  = segmentator,
        sparse_shape        = np.array(grid_size),
    )

    print(f"[Cylinder3D] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle various checkpoint formats
    state_dict = ckpt
    for key in ("model", "state_dict", "model_state_dict"):
        if key in ckpt:
            state_dict = ckpt[key]
            break

    # ── spconv 1.x → 2.x weight layout conversion ────────────────────────
    # The official checkpoint was saved with spconv 1.x which stores sparse
    # conv weights as (kz, ky, kx, in_c, out_c).
    # spconv 2.x expects (out_c, kz, ky, kx, in_c).
    # All 5-D tensors in the state_dict are spconv weights → permute them.
    converted = {}
    n_converted = 0
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor) and v.dim() == 5:
            converted[k] = v.permute(4, 0, 1, 2, 3).contiguous()
            n_converted += 1
        else:
            converted[k] = v
    if n_converted:
        print(f"[Cylinder3D] Converted {n_converted} spconv v1→v2 weight tensors")
    state_dict = converted

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[Cylinder3D] Missing keys ({len(missing)}): {missing[:5]} ...")
    if unexpected:
        print(f"[Cylinder3D] Unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")

    model.to(device)
    model.eval()
    print(f"[Cylinder3D] Model ready on {device}")
    return model
