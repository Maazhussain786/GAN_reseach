# test_cuda_setup.py
import torch
import sys
import os

print("="*60)
print("CUDA Setup Diagnostics")
print("="*60)

# PyTorch info
print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version (PyTorch): {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")

# System CUDA
cuda_path = os.environ.get('CUDA_PATH', 'Not set')
print(f"\nCUDA_PATH: {cuda_path}")

# Check if nvcc is available
import subprocess
try:
    nvcc_version = subprocess.check_output(['nvcc', '--version']).decode()
    print(f"\nnvcc found:")
    print(nvcc_version)
except:
    print("\n⚠️  nvcc not found in PATH")

# Check compiler
try:
    cl_version = subprocess.check_output(['cl'], stderr=subprocess.STDOUT).decode()
    print("\nMSVC cl.exe found")
except:
    print("\n⚠️  cl.exe not found in PATH")

# Try compiling a test extension
print("\n" + "="*60)
print("Testing PyTorch Extension Compilation")
print("="*60)

from torch.utils.cpp_extension import load_inline

cuda_source = '''
__global__ void test_kernel(float* x) { x[0] = 42.0; }

torch::Tensor test_function(torch::Tensor x) {
    auto x_ptr = x.data_ptr<float>();
    test_kernel<<<1, 1>>>(x_ptr);
    return x;
}
'''

cpp_source = 'torch::Tensor test_function(torch::Tensor x);'

try:
    module = load_inline(
        name='test_extension',
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=['test_function'],
        verbose=True
    )
    print("\n✅ SUCCESS: CUDA extension compiled!")
    
    test_tensor = torch.zeros(1, device='cuda')
    result = module.test_function(test_tensor)
    print(f"✅ Extension executed: {result.item()}")
    
except Exception as e:
    print(f"\n❌ FAILED: {e}")

print("\n" + "="*60)