#!/usr/bin/env python3
"""
Setup script for mental health monitoring system.

This script helps users get started quickly by setting up the environment,
installing dependencies, and running initial tests.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
import argparse


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False


def check_python_version() -> bool:
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not compatible")
        print("Please use Python 3.10 or higher")
        return False


def install_dependencies() -> bool:
    """Install project dependencies."""
    print("\n📦 Installing dependencies...")
    
    # Check if requirements.txt exists
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found")
        return False
    
    # Install dependencies
    return run_command("pip install -r requirements.txt", "Installing Python packages")


def setup_pre_commit() -> bool:
    """Setup pre-commit hooks."""
    print("\n🔧 Setting up pre-commit hooks...")
    
    # Install pre-commit
    if not run_command("pip install pre-commit", "Installing pre-commit"):
        return False
    
    # Install hooks
    return run_command("pre-commit install", "Installing pre-commit hooks")


def run_tests() -> bool:
    """Run test suite."""
    print("\n🧪 Running tests...")
    
    # Check if pytest is available
    try:
        import pytest
    except ImportError:
        print("Installing pytest...")
        if not run_command("pip install pytest", "Installing pytest"):
            return False
    
    # Run tests
    return run_command("python -m pytest tests/ -v", "Running test suite")


def create_directories() -> bool:
    """Create necessary directories."""
    print("\n📁 Creating directories...")
    
    directories = [
        "assets",
        "checkpoints", 
        "logs",
        "data",
        "notebooks"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    return True


def generate_sample_data() -> bool:
    """Generate sample data for testing."""
    print("\n📊 Generating sample data...")
    
    try:
        # Add src to path
        sys.path.append(str(Path(__file__).parent / "src"))
        
        from data import MentalHealthDataProcessor
        
        # Create sample data
        config = {
            "synthetic_data_size": 100,
            "min_text_length": 10,
            "max_text_length": 512,
            "train_split": 0.7,
            "val_split": 0.15,
            "test_split": 0.15
        }
        
        processor = MentalHealthDataProcessor(config)
        sample_data = processor.generate_synthetic_data(100)
        
        # Save sample data
        import json
        with open("data/sample_data.json", "w") as f:
            json.dump(sample_data, f, indent=2)
        
        print(f"✅ Generated {len(sample_data)} sample data points")
        return True
        
    except Exception as e:
        print(f"❌ Failed to generate sample data: {e}")
        return False


def run_demo_check() -> bool:
    """Check if demo can be started."""
    print("\n🎮 Checking demo setup...")
    
    try:
        # Check if streamlit is available
        import streamlit
        print("✅ Streamlit is available")
        
        # Check if demo file exists
        if Path("demo/app.py").exists():
            print("✅ Demo application found")
            return True
        else:
            print("❌ Demo application not found")
            return False
            
    except ImportError:
        print("❌ Streamlit not installed")
        return False


def print_next_steps():
    """Print next steps for the user."""
    print("\n" + "="*80)
    print("🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    print("\n📋 NEXT STEPS:")
    print("1. Train a model:")
    print("   python train.py")
    
    print("\n2. Run the interactive demo:")
    print("   streamlit run demo/app.py")
    
    print("\n3. Evaluate a trained model:")
    print("   python scripts/evaluate.py --checkpoint checkpoints/best_model.pt")
    
    print("\n4. Run tests:")
    print("   python -m pytest tests/ -v")
    
    print("\n5. Format code:")
    print("   black src/ tests/")
    
    print("\n6. Lint code:")
    print("   ruff src/ tests/")
    
    print("\n📚 DOCUMENTATION:")
    print("- README.md: Complete project documentation")
    print("- DISCLAIMER.md: Important usage disclaimers")
    print("- configs/default.yaml: Configuration options")
    
    print("\n⚠️  IMPORTANT REMINDERS:")
    print("- This is a RESEARCH tool only")
    print("- NOT for clinical use or medical diagnosis")
    print("- Always consult healthcare professionals for mental health concerns")
    print("- Use de-identification for sensitive data")
    
    print("\n" + "="*80)


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup mental health monitoring system")
    parser.add_argument("--skip-tests", action="store_true", help="Skip running tests")
    parser.add_argument("--skip-pre-commit", action="store_true", help="Skip pre-commit setup")
    parser.add_argument("--skip-demo", action="store_true", help="Skip demo check")
    
    args = parser.parse_args()
    
    print("🧠 Mental Health Monitoring System Setup")
    print("="*50)
    
    success = True
    
    # Check Python version
    if not check_python_version():
        success = False
    
    # Create directories
    if not create_directories():
        success = False
    
    # Install dependencies
    if not install_dependencies():
        success = False
    
    # Setup pre-commit (optional)
    if not args.skip_pre_commit:
        if not setup_pre_commit():
            print("⚠️  Pre-commit setup failed, but continuing...")
    
    # Generate sample data
    if not generate_sample_data():
        success = False
    
    # Run tests (optional)
    if not args.skip_tests:
        if not run_tests():
            print("⚠️  Tests failed, but continuing...")
    
    # Check demo (optional)
    if not args.skip_demo:
        if not run_demo_check():
            print("⚠️  Demo check failed, but continuing...")
    
    if success:
        print_next_steps()
    else:
        print("\n❌ Setup completed with some issues.")
        print("Please check the error messages above and resolve them.")
        sys.exit(1)


if __name__ == "__main__":
    main()
