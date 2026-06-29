"""
VCD as Supervised Loss - Simple Frame Delta Version

This computes temporal consistency loss by comparing frame-to-frame changes
in LATENT SPACE between predictions and ground truth.
"""

import torch
import torch.nn.functional as F
from einops import rearrange

class SimpleVCDLoss:
    """
    Simple temporal consistency loss based on frame deltas.
    Operates in latent space (no VGG/FFT).
    """
    
    def __init__(self, loss_type: str = "l1"):
        """
        Args:
            loss_type: "l1" or "l2" for absolute or squared error
        """
        self.loss_type = loss_type.lower()
        if self.loss_type not in {"l1", "l2"}:
            raise ValueError(f"loss_type must be 'l1' or 'l2', got {self.loss_type}")
    
    def compute_loss(
        self, 
        pred_video: torch.Tensor,      # (B*T, C, H, W) predicted latents
        target_video: torch.Tensor,    # (B*T, C, H, W) ground truth latents
        num_video_frames: int,         # T
    ) -> torch.Tensor:
        """
        Compute VCD loss as frame-to-frame delta matching.
        
        Args:
            pred_video: Predicted video latents (B*T, C, H, W)
            target_video: Ground truth video latents (B*T, C, H, W)
            num_video_frames: Number of frames per sequence (T)
        
        Returns:
            vcd_loss: Scalar loss value
        """
        # Validate inputs
        if pred_video.shape[0] % num_video_frames != 0:
            raise ValueError(
                f"pred_video batch dim {pred_video.shape[0]} not divisible by "
                f"num_video_frames {num_video_frames}"
            )
        
        if num_video_frames < 2:
            # Need at least 2 frames to compute deltas
            return pred_video.new_zeros(())
        
        batch_size = pred_video.shape[0] // num_video_frames
        
        # Reshape: (B*T, C, H, W) -> (B, T, C, H, W)
        pred_bt = rearrange(
            pred_video, 
            "(b t) c h w -> b t c h w", 
            b=batch_size, 
            t=num_video_frames
        )
        target_bt = rearrange(
            target_video, 
            "(b t) c h w -> b t c h w", 
            b=batch_size, 
            t=num_video_frames
        )
        
        # Compute frame-to-frame deltas
        # delta[i] = frame[i+1] - frame[i]
        pred_delta = pred_bt[:, 1:] - pred_bt[:, :-1]      # (B, T-1, C, H, W)
        target_delta = target_bt[:, 1:] - target_bt[:, :-1]  # (B, T-1, C, H, W)
        
        # Compute error between predicted and target deltas
        delta_error = pred_delta - target_delta
        
        # Apply loss function
        if self.loss_type == "l1":
            loss = delta_error.abs().mean()
        else:  # l2
            loss = delta_error.pow(2).mean()
        
        return loss

