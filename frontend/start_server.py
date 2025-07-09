#!/usr/bin/env python3
"""
EasySteer Server Quick Start Script

This script provides a simple way to start the EasySteer server,
including dependency checks, environment validation, and other features.
"""

import os
import sys
import subprocess
import importlib.util
from pathlib import Path

def check_dependencies():
    """Check for necessary dependencies"""
    required_packages = [
        'flask',
        'flask_cors',
        'transformers',
        'torch',
        'vllm',
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            spec = importlib.util.find_spec(package)
            if spec is None:
                missing_packages.append(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ All required packages are installed")
    return True

def check_environment():
    """Check the environment configuration"""
    print("🔍 Checking environment...")
    
    # Check CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"✅ CUDA available with {gpu_count} GPU(s)")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                print(f"   GPU {i}: {gpu_name}")
        else:
            print("⚠️  CUDA not available - training will use CPU (slower)")
    except ImportError:
        print("⚠️  PyTorch not installed - cannot check CUDA")
    
    # Check working directory
    current_dir = Path.cwd()
    if current_dir.name != 'frontend':
        print(f"⚠️  Current directory: {current_dir}")
        print("💡 Recommended to run from the 'frontend' directory")
    
    # Check results directory
    results_dir = Path("./results")
    if not results_dir.exists():
        print("📁 Creating results directory...")
        results_dir.mkdir(exist_ok=True)
    
    return True

def display_startup_info():
    """Display startup information"""
    print("\n" + "="*60)
    print("🚀 EasySteer - Steer Vector Training & Inference Platform")
    print("="*60)
    print()
    print("📖 Features:")
    print("   • 🎯 Inference with pre-trained steer vectors")
    print("   • 🎓 Train custom steer vectors with ReFT")
    print("   • 🌐 Multi-language support (中文/English)")
    print("   • 📊 Real-time training progress monitoring")
    print("   • 🎨 Beautiful web interface")
    print()
    print("🔗 Access URLs:")
    print("   • Web Interface: http://localhost:5000")
    print("   • API Health: http://localhost:5000/api/health")
    print()
    print("📝 Quick Start:")
    print("   1. Open http://localhost:5000 in your browser")
    print("   2. Choose 'Inference' or 'Training' tab")
    print("   3. Fill in the configuration and start!")
    print()
    print("💡 Demo Script:")
    print("   python demo_training.py --model /path/to/model --preset emoji")
    print()

def main():
    """Main function"""
    print("🎯 EasySteer Server Launcher")
    print("-" * 30)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Dependency check failed. Please install missing packages.")
        sys.exit(1)
    
    # Check environment
    if not check_environment():
        print("\n❌ Environment check failed.")
        sys.exit(1)
    
    # Display startup info
    display_startup_info()
    
    # Ask whether to continue
    try:
        response = input("Press Enter to start the server (or Ctrl+C to cancel): ")
    except KeyboardInterrupt:
        print("\n👋 Startup cancelled by user.")
        sys.exit(0)
    
    # Start the server
    print("\n🚀 Starting EasySteer server...")
    print("=" * 40)
    
    try:
        # Set environment variables
        env = os.environ.copy()
        env['FLASK_ENV'] = 'development'
        env['VLLM_USE_V1'] = '0'  # Ensure V0 is used to support steer vectors
        
        # Start the Flask application
        subprocess.run([sys.executable, 'app.py'], env=env)
        
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user.")
    except FileNotFoundError:
        print("\n❌ app.py not found. Please run this script from the frontend directory.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error starting server: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 