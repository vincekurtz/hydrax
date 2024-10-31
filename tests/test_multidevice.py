import os

# Use only CPU, which gives us the opportunity to treat each CPU core as a
# device, and avoid the need for multiple GPUs.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax

if __name__ == "__main__":
    print(jax.devices())
