import torch
import time

print("--- Intel Arc GPU AI Pipeline Test ---")
# 1. Check if the native Intel XPU engine is found
gpu_available = torch.xpu.is_available()
print(f"Intel GPU Acceleration Available: {gpu_available}")

if gpu_available:
    # 2. Query the device properties
    device_name = torch.xpu.get_device_name(0)
    print(f"Active AI Hardware: {device_name}\n")
    
    # 3. Allocating tensors inside the Intel Arc VRAM
    print("Initializing matrix tensors inside GPU memory...")
    device = torch.device("xpu:0")
    matrix_a = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)
    matrix_b = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)
    #matrix_a = torch.randn(4096, 4096, device=device, dtype=torch.float32)
    #matrix_b = torch.randn(4096, 4096, device=device, dtype=torch.float32)


    # 4. Computation benchmark loop
    print("Executing floating-point multiplication loop on Intel Arc hardware...")
    start_time = time.time()
    for _ in range(50):
        result = torch.matmul(matrix_a, matrix_b)
    
    # 5. Flush queue to sync time clock accuracy
    torch.xpu.synchronize()
    end_time = time.time()
    
    print("\n✅ Verification Successful!")
    print(f"Matrix multiplication completed in: {end_time - start_time:.4f} seconds.")
else:
    print("\n❌ Error: PyTorch cannot communicate with the Intel XPU backend.")
