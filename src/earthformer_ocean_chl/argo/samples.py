"""Validated transfer-sample writing."""
from pathlib import Path
import numpy as np
def validate_sample_shapes(inputs, targets, expected_input=(14,45,60,6), expected_target=(14,45,60,20)):
    if inputs.shape[0] != targets.shape[0]: raise ValueError("Input and target sample counts differ")
    if tuple(inputs.shape[1:]) != tuple(expected_input): raise ValueError(f"Input sample shape must be {expected_input}")
    if tuple(targets.shape[1:]) != tuple(expected_target): raise ValueError(f"Target sample shape must be {expected_target}")
def write_transfer_samples(inputs, targets, output_root):
    validate_sample_shapes(inputs, targets); root=Path(output_root); ip=root/'transfer_input'; op=root/'transfer_output'; ip.mkdir(parents=True,exist_ok=True); op.mkdir(parents=True,exist_ok=True)
    for i,(x,y) in enumerate(zip(inputs,targets)): np.save(ip/f'input_{i}.npy',x); np.save(op/f'output_{i}.npy',y)
    return len(inputs)
