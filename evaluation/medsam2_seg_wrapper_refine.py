
import os
import time
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from dino_unet import DINOv3_S_UNet


def _mask_to_boundary_points(mask_bin: np.ndarray, max_points: int = 30, seed: Optional[int] = None) -> np.ndarray:
    """
    Extract boundary points from a binary mask using a fast erosion-based boundary:
        boundary = mask XOR erode(mask)

    Implement erosion via torch maxpool to avoid extra deps.

    Args:
        mask_bin: (H,W) uint8/bool in {0,1}
        max_points: number of boundary points to sample
        seed: optional rng seed

    Returns:
        (N,2) float32 array of point coords in pixel space: [[x,y], ...]
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    if mask_bin.ndim != 2:
        raise ValueError(f"mask_bin must be 2D, got shape {mask_bin.shape}")

    H, W = mask_bin.shape
    if mask_bin.sum() == 0:
        return np.zeros((0, 2), dtype=np.float32)

    # torch erosion: erode(mask) ≈ 1 - dilate(1-mask)
    m = torch.from_numpy(mask_bin.astype(np.float32))[None, None, ...]  # 1x1xHxW
    inv = 1.0 - m
    dil_inv = F.max_pool2d(inv, kernel_size=3, stride=1, padding=1)
    eroded = (1.0 - dil_inv).clamp(0, 1)
    boundary = (m > 0.5) ^ (eroded > 0.5)
    boundary_np = boundary.squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)

    ys, xs = np.where(boundary_np > 0)
    if len(xs) == 0:
        # fallback: sample from foreground pixels if boundary degenerates
        ys, xs = np.where(mask_bin > 0)

    n = min(max_points, len(xs))
    if n <= 0:
        return np.zeros((0, 2), dtype=np.float32)

    idx = rng.choice(len(xs), size=n, replace=False) if len(xs) > n else np.arange(len(xs))
    pts = np.stack([xs[idx], ys[idx]], axis=1).astype(np.float32)
    return pts


def _mask_to_box(mask_bin: np.ndarray, pad: int = 4) -> Optional[np.ndarray]:
    """
    Compute tight box around mask with padding, in pixel coordinates [x0,y0,x1,y1].
    Returns None if mask is empty.
    """
    if mask_bin.sum() == 0:
        return None
    ys, xs = np.where(mask_bin > 0)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    H, W = mask_bin.shape
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(W - 1, x1 + pad)
    y1 = min(H - 1, y1 + pad)
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def _sample_background_points_outside_box(
    box_xyxy: np.ndarray,
    H: int,
    W: int,
    max_points: int = 8,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Sample background points outside a given box. If not enough space, fall back to corners.
    Returns (N,2) in pixel coords [[x,y], ...]
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    x0, y0, x1, y1 = box_xyxy.tolist()

    # define a mask of "outside box" and sample uniformly from it
    outside = np.ones((H, W), dtype=np.uint8)
    outside[int(y0): int(y1) + 1, int(x0): int(x1) + 1] = 0
    ys, xs = np.where(outside > 0)

    if len(xs) == 0:
        # fallback to corners
        corners = np.array([[0, 0], [W - 1, 0], [0, H - 1], [W - 1, H - 1]], dtype=np.float32)
        n = min(max_points, len(corners))
        return corners[:n]

    n = min(max_points, len(xs))
    idx = rng.choice(len(xs), size=n, replace=False) if len(xs) > n else np.arange(len(xs))
    pts = np.stack([xs[idx], ys[idx]], axis=1).astype(np.float32)
    return pts


class MedSAM2SegWrapperRefine:
    """
    A practical wrapper that uses DINO-UNet to get an initial mask, then uses MedSAM2/SAM2 to refine it.

    Key changes vs naive usage:
      - Use boundary points from the initial mask (strong constraint)
      - Use box prompt from the initial mask
      - Use normalize_coords=True (safer for SAM2 internal scaling)
      - Optionally use mask_input for refinement (if SAM2 predictor supports it)
    """

    def __init__(
        self,
        sam2_cfg: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        dino_unet_ckpt: Optional[str] = None,
        # prompt config
        fg_points: int = 30,
        bg_points: int = 8,
        box_pad: int = 4,
        # refinement config
        enable_mask_prompt: bool = False,
        # debug
        debug: bool = False,
        debug_dir: str = "plot_refine",
        seed: Optional[int] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.debug = debug
        self.debug_dir = debug_dir
        self.seed = seed

        self.fg_points = int(fg_points)
        self.bg_points = int(bg_points)
        self.box_pad = int(box_pad)
        self.enable_mask_prompt = bool(enable_mask_prompt)

        # Load MedSAM2/SAM2
        self.sam2_model = build_sam2(
            config_file=sam2_cfg,
            checkpoint_path=checkpoint_path,
            device=self.device,
            mode="eval",
        )
        self.predictor = SAM2ImagePredictor(self.sam2_model)

        # Load DINO-UNet
        self.dino_unet_model = None
        if dino_unet_ckpt:
            self.dino_unet_model = DINOv3_S_UNet()
            state = torch.load(dino_unet_ckpt, map_location=self.device)
            self.dino_unet_model.load_state_dict(state)
            self.dino_unet_model.to(self.device).eval()

        if self.debug:
            os.makedirs(self.debug_dir, exist_ok=True)

    @torch.no_grad()
    def __call__(self, image: torch.Tensor, batch: Optional[dict] = None) -> torch.Tensor:
        """
        Args:
            image: (B,C,H,W) - unused; kept for compatibility with existing pipelines
            batch: dict containing at least:
                - "image_rgb": (B,3,H,W) tensor for visualization (optional)
                - "image": (B,C,H,W) tensor used for DINO input & conversion to uint8 for SAM set_image
                - "label": (B,1,H,W) tensor ground truth (optional; only for debug print)

        Returns:
            mask_pred: (B,1,H,W) float tensor in {0,1}
        """
        if batch is None:
            raise ValueError("batch is required and must contain 'image' at minimum.")

        img_batch = batch["image"]  # (B,C,H,W)
        B = img_batch.shape[0]
        results: List[torch.Tensor] = []

        for i in range(B):
            img_processed = img_batch[i]  # (C,H,W), assumed 0..1
            image_rgb = batch["image_rgb"][i]
            img_np = (img_processed.permute(1, 2, 0).detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            H, W = img_np.shape[:2]

            # set image for SAM2 predictor
            self.predictor.set_image(img_np)

            # ---- 1) initial mask from DINO (or fallback to center point only) ----
            dino_bin = None
            if self.dino_unet_model is not None:
                dino_input = img_processed.unsqueeze(0).to(self.device)
                dino_logits = self.dino_unet_model(dino_input).squeeze(0).squeeze(0)
                dino_prob = torch.sigmoid(dino_logits).detach().cpu().numpy().astype(np.float32)
                dino_bin = (dino_prob > 0.5).astype(np.uint8)

            # If no DINO, we will use center FG point only (not recommended)
            if dino_bin is None or dino_bin.sum() == 0:
                point_coords = np.array([[W / 2.0, H / 2.0]], dtype=np.float32)
                point_labels = np.array([1], dtype=np.int32)
                box = None
                mask_input = None
            else:
                # ---- 2) build strong prompts from the initial mask ----
                fg_pts = _mask_to_boundary_points(dino_bin, max_points=self.fg_points, seed=self.seed)
                box_xyxy = _mask_to_box(dino_bin, pad=self.box_pad)
                box = None
                if box_xyxy is not None:
                    box = box_xyxy[None, :]  # (1,4)

                bg_pts = None
                if box_xyxy is not None and self.bg_points > 0:
                    bg_pts = _sample_background_points_outside_box(
                        box_xyxy, H=H, W=W, max_points=self.bg_points, seed=self.seed
                    )

                if fg_pts.shape[0] == 0:
                    # fallback: center point in box
                    if box_xyxy is not None:
                        cx = (box_xyxy[0] + box_xyxy[2]) / 2.0
                        cy = (box_xyxy[1] + box_xyxy[3]) / 2.0
                    else:
                        cx, cy = W / 2.0, H / 2.0
                    fg_pts = np.array([[cx, cy]], dtype=np.float32)

                if bg_pts is None or bg_pts.shape[0] == 0:
                    # small fallback set
                    bg_pts = np.array([[0, 0], [W - 1, 0], [0, H - 1], [W - 1, H - 1]], dtype=np.float32)[: min(4, self.bg_points)]

                point_coords = np.concatenate([fg_pts, bg_pts], axis=0).astype(np.float32)
                point_labels = np.concatenate(
                    [np.ones((fg_pts.shape[0],), dtype=np.int32), np.zeros((bg_pts.shape[0],), dtype=np.int32)],
                    axis=0,
                )

                # Optional mask prompt (if supported by predictor)
                mask_input = dino_bin[None, :, :].astype(np.float32) if self.enable_mask_prompt else None

            # ---- 3) predict ----
            # 注意：normalize_coords=True 时，SAM2 内部会自动将像素坐标归一化
            # 因此直接传入原始像素坐标即可，不要手动归一化（否则会 double normalization）
            predict_kwargs = dict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=False,
                return_logits=False,
                normalize_coords=True,
            )
            if box is not None:
                predict_kwargs["box"] = box

            # mask prompt is optional & may not be supported depending on SAM2ImagePredictor version
            if mask_input is not None:
                try:
                    predict_kwargs["mask_input"] = mask_input
                except Exception:
                    pass

            masks, scores, _ = self.predictor.predict(**predict_kwargs)
            best_mask = masks[int(np.argmax(scores))].astype(np.float32)  # (H,W) in {0,1}
            best_mask_bin = (best_mask > 0).astype(np.uint8)
            import os
            from PIL import Image
            plot_dir = "plot"
            os.makedirs(plot_dir, exist_ok=True)
            # 使用时间戳区分文件名
            import time
            ts = int(time.time())
            if self.debug:
                ts = int(time.time() * 1000)
                # save debug masks
                from PIL import Image
                Image.fromarray((best_mask > 0.5).astype(np.uint8) * 255).save(
                    os.path.join(self.debug_dir, f"sam_refine_mask_{i}_{ts}.png")
                )
                if dino_bin is not None:
                    Image.fromarray(dino_bin.astype(np.uint8) * 255).save(
                        os.path.join(self.debug_dir, f"dino_mask_{i}_{ts}.png")
                    )

                # optional metric print if label exists
                if "label" in batch:
                    label_map = batch["label"][i][0].detach().cpu().numpy()
                    label_bin = (label_map > 0.5).astype(np.uint8)
                    inter = np.logical_and(best_mask > 0.5, label_bin > 0).sum()
                    dice = 2 * inter / ((best_mask > 0.5).sum() + label_bin.sum() + 1e-6)
                    print(f"[Debug][SAM2 refine vs Label] Sample {i}: Dice={dice:.4f}, PredSum={(best_mask>0.5).sum()}, LabelSum={label_bin.sum()}")

            mask_tensor = torch.from_numpy(best_mask).float().unsqueeze(0).to(self.device)  # (1,H,W)
            results.append(mask_tensor)

        return torch.stack(results, dim=0)  # (B,1,H,W)

    def eval(self):
        self.sam2_model.eval()
        if self.dino_unet_model is not None:
            self.dino_unet_model.eval()
        return self

    def to(self, device):
        self.device = device
        self.sam2_model = self.sam2_model.to(device)
        self.predictor = SAM2ImagePredictor(self.sam2_model)
        if self.dino_unet_model is not None:
            self.dino_unet_model = self.dino_unet_model.to(device)
        return self
