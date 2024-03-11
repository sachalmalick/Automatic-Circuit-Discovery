import torch
import time

if torch.cuda.is_available():
    cuda_device = torch.device("cuda:0")
    print("Using CUDA device:", torch.cuda.get_device_name(cuda_device))
else:
    raise SystemExit("CUDA is not available. This script requires a GPU.")

size = 10000
a = torch.rand(size, size, device=cuda_device)
b = torch.rand(size, size, device=cuda_device)

start_time = time.time()
for _ in range(1000):
    result = torch.matmul(a, b)
end_time = time.time()

print(f"Completed 1000 matrix multiplications in {end_time - start_time:.2f} seconds")
