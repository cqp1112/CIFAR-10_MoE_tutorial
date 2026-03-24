import torch

print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version in torch:", torch.version.cuda)
print("device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("device name:", torch.cuda.get_device_name(0))
    x = torch.randn(3, 3, device="cuda")
    print("tensor device:", x.device)
    print(x)
else:
    print("GPU not available, fallback to CPU")