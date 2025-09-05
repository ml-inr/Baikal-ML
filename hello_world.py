"""Test script to verify environment setup."""

import sys
from pathlib import Path

def test_imports():
    """Test all critical imports work."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU count: {torch.cuda.device_count()}")
            print(f"   Current GPU: {torch.cuda.get_device_name()}")
        
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
        
        import h5py
        print(f"✅ HDF5 {h5py.__version__}")
        
        import yaml
        print("✅ PyYAML")
        
        import pandas as pd
        print(f"✅ Pandas {pd.__version__}")
        
        import sklearn
        print(f"✅ Scikit-learn {sklearn.__version__}")
        
        import matplotlib
        print(f"✅ Matplotlib {matplotlib.__version__}")
        
        import ROOT
        print(f"✅ PyROOT {ROOT.gROOT.GetVersion()}")
        
        import awkward as ak
        print(f"✅ Awkward {ak.__version__}")
        
        try:
            import clearml
            print(f"✅ ClearML {clearml.__version__}")
        except ImportError:
            print("⚠️  ClearML not available")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    return True

def test_cuda_functionality():
    """Test CUDA functionality specifically."""
    print("\nTesting CUDA functionality...")
    
    import torch
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available - running on CPU")
        return True
    
    # Test GPU tensor operations
    device = torch.device("cuda")
    x = torch.randn(100, 100, device=device)
    y = torch.randn(100, 100, device=device)
    z = torch.mm(x, y)
    print(f"✅ GPU tensor operations: {z.shape} on {z.device}")
    
    # Test GPU model
    model = torch.nn.Linear(100, 50).to(device)
    input_data = torch.randn(32, 100, device=device)
    output = model(input_data)
    print(f"✅ GPU model inference: {output.shape} on {output.device}")
    
    return True

def test_basic_functionality():
    """Test basic PyTorch functionality."""
    print("\nTesting basic functionality...")
    
    import torch
    
    # Test tensor operations
    x = torch.randn(3, 4)
    y = torch.randn(4, 5)
    z = torch.mm(x, y)
    print(f"✅ CPU tensor operations: {z.shape}")
    
    # Test simple model
    model = torch.nn.Linear(10, 1)
    input_data = torch.randn(5, 10)
    output = model(input_data)
    print(f"✅ CPU model inference: {output.shape}")
    
    return True

def main():
    """Main test function."""
    print("🧪 Testing Baikal25 Environment Setup")
    print("=" * 50)
    
    print(f"Python version: {sys.version}")
    print(f"Current directory: {Path.cwd()}")
    
    # Test imports
    if not test_imports():
        print("\n❌ Environment setup failed!")
        sys.exit(1)
    
    # Test basic functionality
    if not test_basic_functionality():
        print("\n❌ Basic functionality test failed!")
        sys.exit(1)
    
    # Test CUDA functionality
    if not test_cuda_functionality():
        print("\n❌ CUDA functionality test failed!")
        sys.exit(1)
    
    print("\n🎉 Baikal25 environment setup successful!")
    print("Ready to start neural network research with CUDA 12.6 support!")

if __name__ == "__main__":
    main()