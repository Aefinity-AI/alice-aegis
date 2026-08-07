import platform, sys
print("aefinity kaggle lane smoke: python", sys.version.split()[0], platform.machine())
try:
    import torch
    print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
except Exception as e:
    print("torch import failed:", e)
print("SMOKE_OK")
