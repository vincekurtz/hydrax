#!/bin/bash
uv run examples/humanoid_mocap_headless.py \
    --warp \
    --duration 10.0 \
    --risk "average" \
    --reference_filename "Lafan1/mocap/UnitreeG1/walk1_subject1.npz"
