import subprocess
import sys

    
def install_dependecies():
    dependencies = [
        "scikit-learn",
        "tensorflow",
        "matplotlib",
        "numpy"
    ]

    for package in dependencies:
        try:
            __import__(package.split('-')[0])
            print(f"{package} is already installed.")
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])