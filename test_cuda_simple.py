# test_cuda_simple.py
# Quick test of PyTorch CUDA extension compilation

import torch
import sys

print("="*70)
print("PyTorch CUDA Extension Test")
print("="*70)

print(f"\nPyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

# Check Ninja
try:
    import ninja
    print(f"✅ Ninja installed: {ninja.__version__}")
except ImportError:
    print("❌ Ninja NOT installed - run: pip install ninja")
    sys.exit(1)

# Test compilation
print("\n" + "="*70)
print("Attempting to compile test CUDA extension...")
print("="*70)

from torch.utils.cpp_extension import load_inline

cuda_source = '''
__global__ void add_kernel(float* out, const float* a, const float* b, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

torch::Tensor add_tensors(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);
    
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    
    add_kernel<<<blocks, threads>>>(
        out.data_ptr<float>(),
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        size
    );
    
    return out;
}
'''

cpp_source = 'torch::Tensor add_tensors(torch::Tensor a, torch::Tensor b);'

try:
    print("\nCompiling... (this may take 30-60 seconds)")
    print("You should see compilation output below:\n")
    
    module = load_inline(
        name='cuda_test_extension',
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=['add_tensors'],
        verbose=True,
        extra_cuda_cflags=['--use_fast_math']
    )
    
    print("\n" + "="*70)
    print("✅ COMPILATION SUCCESSFUL!")
    print("="*70)
    
    # Test the extension
    print("\nTesting compiled extension...")
    a = torch.randn(1000, device='cuda')
    b = torch.randn(1000, device='cuda')
    
    result = module.add_tensors(a, b)
    expected = a + b
    
    diff = (result - expected).abs().max().item()
    
    if diff < 1e-5:
        print(f"✅ Extension works correctly! (max diff: {diff:.2e})")
        print("\n" + "="*70)
        print("🎉 YOUR SYSTEM IS READY FOR STYLEGAN2!")
        print("="*70)
    else:
        print(f"⚠️  Extension compiled but results differ: {diff}")
        
except Exception as e:
    print("\n" + "="*70)
    print("❌ COMPILATION FAILED")
    print("="*70)
    print(f"\nError: {e}\n")
    
    print("Common fixes:")
    print("1. Make sure you're running from 'Developer PowerShell for VS 2022'")
    print("2. Or run: & 'C:\\Program Files\\Microsoft Visual Studio\\2022\\BuildTools\\VC\\Auxiliary\\Build\\vcvars64.bat'")
    print("3. Verify 'cl' command works (MSVC compiler)")
    print("4. Install Ninja: pip install ninja")
    
    import traceback
    print("\nFull traceback:")
    traceback.print_exc()