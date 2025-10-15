import torch
import sys
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Test basic tensor operations
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(10, 10).to(device)
    y = torch.randn(10, 10).to(device)
    z = torch.matmul(x, y)
    print(f"✅ Basic tensor operations work on {device}")
except Exception as e:
    print(f"❌ Error with tensor operations: {e}")

# Test transformers import
try:
    from transformers import AutoTokenizer
    print("✅ Transformers library imported successfully")
except Exception as e:
    print(f"❌ Error importing transformers: {e}")

# Test timm import
try:
    import timm
    print("✅ Timm library imported successfully")
except Exception as e:
    print(f"❌ Error importing timm: {e}")

print("\nBasic setup test completed!")