import os
from pathlib import Path

# package root
ROOT = str(Path(__file__).parent.absolute())

# Set XLA flags for better performance
os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=true "
