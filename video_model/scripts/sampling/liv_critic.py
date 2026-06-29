"""LIV reward critic for scoring generated video frames against language.

Reuses the LIV model already loaded inside DiffusionEngine
(see sgm/models/diffusion.py: self.liv_model + self._cr_score).
"""
import numpy as np
import torch


class LIVCritic:
    def __init__(self, diffusion_model, device="cuda"):
        if getattr(diffusion_model, "liv_model", None) is None:
            raise RuntimeError(
                "Diffusion model has no liv_model. "
                "Ensure LIV is importable (pip install LIV-robotics) "
                "and the engine was built with LIV available."
            )
        self.model = diffusion_model
        self.device = torch.device(device)
        self.model._cr_move_to(self.device)

    @torch.no_grad()
    def score_frames_uint8(self, frames_uint8, caption):
        """frames_uint8: (T, H, W, 3) np.uint8 RGB. Returns (T,) float numpy."""
        if len(frames_uint8) == 0:
            return np.zeros((0,), dtype=np.float32)
        x = torch.from_numpy(np.ascontiguousarray(frames_uint8)).float() / 255.0
        x = x.permute(0, 3, 1, 2).to(self.device)
        caps = [str(caption)] * x.shape[0]
        scores = self.model._cr_score(x, caps)
        return scores.detach().float().cpu().numpy()

    @torch.no_grad()
    def score_against_many(self, frames_uint8, captions):
        """Returns (T, K) cosine sims for K captions. Loops captions to keep it simple."""
        if len(frames_uint8) == 0 or len(captions) == 0:
            return np.zeros((len(frames_uint8), len(captions)), dtype=np.float32)
        out = np.stack(
            [self.score_frames_uint8(frames_uint8, c) for c in captions], axis=1
        )
        return out
