"""Small helpers shared by the train / predict / benchmark scripts."""
import torch

from model import build_model


def select_device(name="auto"):
    """Pick the compute device.

    "auto" uses the GPU only if a tensor can really be placed on it:
    torch.cuda.is_available() also returns True on machines where the NVIDIA
    driver is too old for the installed PyTorch build, and the first real CUDA
    call then fails. In that case we fall back to the CPU.
    """
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda")
            return torch.device("cuda")
        except RuntimeError as err:
            print(f"CUDA is present but not usable ({str(err).splitlines()[0]}); using the CPU instead.")
    return torch.device("cpu")


def load_model(checkpoint_path, device):
    """Rebuild the network stored in a checkpoint. Returns (model in eval mode, checkpoint dict)."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = build_model(checkpoint["model_name"], **checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, checkpoint
