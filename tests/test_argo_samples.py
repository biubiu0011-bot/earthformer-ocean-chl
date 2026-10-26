from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from earthformer_ocean_chl.argo.samples import validate_sample_shapes, write_transfer_samples
def test_transfer_writer_creates_paired_files(tmp_path):
 x=np.zeros((1,14,45,60,6)); y=np.zeros((1,14,45,60,20)); assert write_transfer_samples(x,y,tmp_path)==1; assert (tmp_path/'transfer_input'/'input_0.npy').exists(); assert (tmp_path/'transfer_output'/'output_0.npy').exists()
def test_transfer_validator_rejects_wrong_channels():
 with pytest.raises(ValueError,match='Input sample shape'): validate_sample_shapes(np.zeros((1,14,45,60,5)),np.zeros((1,14,45,60,20)))
