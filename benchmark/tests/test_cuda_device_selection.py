"""Accept both CUDA aliases without needing GPU hardware in the test."""
from types import SimpleNamespace
from unittest.mock import Mock, patch
import pytest
from benchmark.run import evaluate


@pytest.mark.parametrize('name,index', [('cuda', 0), ('cuda:1', 1)])
def test_cuda_device_alias_reaches_data_loading(name, index):
    class DataReached(Exception):
        pass
    data = Mock()
    data.read_bytes.side_effect = DataReached
    args = SimpleNamespace(device=name, threads=1, data=data)
    with patch('torch.set_num_threads'), patch('torch.set_num_interop_threads'), \
         patch('torch.manual_seed'), patch('torch.cuda.is_available', return_value=True), \
         patch('torch.cuda.current_device', return_value=0), \
         patch('torch.cuda.set_device') as select, patch('torch.cuda.reset_peak_memory_stats'):
        with pytest.raises(DataReached):
            evaluate(args)
        select.assert_called_once_with(index)
