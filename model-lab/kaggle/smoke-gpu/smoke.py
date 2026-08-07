import platform, sys
print("aefinity kaggle lane smoke: python", sys.version.split()[0], platform.machine())
try:
    import torch
    print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device_count", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"gpu{i}: {p.name} {p.total_memory/1e9:.1f}GB CC{p.major}.{p.minor}")
except Exception as e:
    print("torch probe failed:", e)
print("GPU_SMOKE_OK")
