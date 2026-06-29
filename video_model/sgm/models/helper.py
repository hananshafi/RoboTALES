import torch.nn as nn

def _unwrap(m: nn.Module) -> nn.Module:
    while True:
        for attr in ("module", "model", "_orig_mod", "inner_model", "backbone", "policy"):
            if hasattr(m, attr) and isinstance(getattr(m, attr), nn.Module):
                m = getattr(m, attr)
                break
        else:
            return m

def find_videounet(root: nn.Module) -> nn.Module:
    m = _unwrap(root)
    # If the top-level already has input/output blocks, that's our VideoUNet
    if hasattr(m, "input_blocks") and hasattr(m, "output_blocks"):
        return m
    # Common in videopolicy: UNet lives under diffusion_model
    dm = getattr(m, "diffusion_model", None)
    if isinstance(dm, nn.Module) and hasattr(dm, "input_blocks") and hasattr(dm, "output_blocks"):
        return dm
    # Fallback: scan
    for _, mod in m.named_modules():
        if hasattr(mod, "input_blocks") and hasattr(mod, "output_blocks"):
            return mod
    raise RuntimeError("VideoUNet with input_blocks/output_blocks not found")