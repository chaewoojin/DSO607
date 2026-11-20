"""Run imports on the login node; add --gpu only inside a GPU allocation."""
import argparse
import importlib
import sys

p = argparse.ArgumentParser()
p.add_argument("--gpu", action="store_true")
args = p.parse_args()
for name in ["torch", "torchvision", "numpy", "scipy", "sklearn", "pandas", "matplotlib", "joblib"]:
    mod = importlib.import_module(name)
    print(name, getattr(mod, "__version__", "unknown"))
if args.gpu:
    import torch
    if not torch.cuda.is_available():
        sys.exit("CUDA unavailable: check your Slurm allocation and PyTorch module.")
    a = torch.ones((64, 64), device="cuda")
    assert (a @ a)[0, 0].item() == 64
    print("GPU matrix calculation passed:", torch.cuda.get_device_name(0))
