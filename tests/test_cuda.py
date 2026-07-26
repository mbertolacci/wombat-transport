from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from wombat_transport.cuda import CudaRuntime, CudaUnavailableError
from wombat_transport.cuda.history import (
    accumulate_history_sum,
    accumulate_history_sums,
)
from wombat_transport.fields import TracerField


def test_importing_cuda_support_does_not_import_cupy():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import wombat_transport.cuda; "
                "import wombat_transport.transport.pbl._cuda; "
                "assert 'cupy' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def _cuda_runtime_or_skip() -> CudaRuntime:
    try:
        return CudaRuntime()
    except CudaUnavailableError as exc:
        pytest.skip(str(exc))


@pytest.mark.cuda
def test_cuda_runtime_accounts_for_explicit_transfers():
    runtime = _cuda_runtime_or_skip()
    host = np.arange(48, dtype=np.float64).reshape(2, 3, 8)

    device = runtime.to_device(host)
    actual = runtime.to_host(device)

    np.testing.assert_array_equal(actual, host)
    assert runtime.is_device_array(device)
    assert runtime.transfer_stats.host_to_device_count == 1
    assert runtime.transfer_stats.host_to_device_bytes == host.nbytes
    assert runtime.transfer_stats.device_to_host_count == 1
    assert runtime.transfer_stats.device_to_host_bytes == host.nbytes
    assert runtime.transfer_stats.explicit_synchronizations == 0

    runtime.synchronize()
    assert runtime.transfer_stats.explicit_synchronizations == 1


@pytest.mark.cuda
def test_cuda_runtime_detects_device_memory_overlap_without_array_work():
    runtime = _cuda_runtime_or_skip()
    storage = runtime.empty((64,), dtype=np.float64)

    assert runtime.shares_memory(storage, storage)
    assert runtime.shares_memory(storage[8:32], storage[24:40])
    assert not runtime.shares_memory(storage[:16], storage[16:])
    assert not runtime.shares_memory(
        storage,
        runtime.empty((64,), dtype=np.float64),
    )
    with pytest.raises(ValueError, match="C-contiguous"):
        runtime.shares_memory(storage.reshape(8, 8)[:, 0], storage)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_cuda_history_accumulates_resident_tracer_blocks(dtype):
    runtime = _cuda_runtime_or_skip()
    rng = np.random.default_rng(20140901)
    shape = (2, 3, 4, 5, 8)
    host_values = rng.standard_normal(shape).astype(dtype)
    host_delta = rng.standard_normal(shape).astype(dtype)
    expected_sum = np.zeros(shape, dtype=dtype)
    expected_sums = np.zeros((3, *shape), dtype=dtype)
    device_values = runtime.to_device(host_values)
    device_delta = runtime.to_device(host_delta)
    device_sum = runtime.zeros(shape, dtype=dtype)
    device_sums = runtime.zeros((3, *shape), dtype=dtype)

    for _ in range(6):
        accumulate_history_sum(device_sum, device_values)
        accumulate_history_sums(device_sums, device_values)
        expected_sum += host_values
        expected_sums += host_values[None, ...]
        runtime.array_module.add(device_values, device_delta, out=device_values)
        host_values += host_delta

    actual_sum = runtime.to_host(device_sum)
    actual_sums = runtime.to_host(device_sums)

    np.testing.assert_array_equal(actual_sum, expected_sum)
    np.testing.assert_array_equal(actual_sums, expected_sums)
    assert runtime.transfer_stats.host_to_device_count == 2
    assert runtime.transfer_stats.device_to_host_count == 2


@pytest.mark.cuda
def test_cuda_history_accumulates_float32_state_into_float64_sums():
    runtime = _cuda_runtime_or_skip()
    host = np.linspace(-1.0, 1.0, 48, dtype=np.float32).reshape(2, 3, 8)
    values = runtime.to_device(host)
    summed = runtime.zeros(host.shape, dtype=np.float64)

    for _ in range(7):
        accumulate_history_sum(summed, values)

    np.testing.assert_array_equal(
        runtime.to_host(summed),
        host.astype(np.float64) * 7.0,
    )


@pytest.mark.cuda
def test_cuda_history_rejects_shape_and_dtype_mismatches():
    runtime = _cuda_runtime_or_skip()
    values = runtime.zeros((2, 3), dtype=np.float64)

    with pytest.raises(ValueError, match="does not match"):
        accumulate_history_sum(runtime.zeros((3, 2), dtype=np.float64), values)
    with pytest.raises(TypeError, match="dtype"):
        accumulate_history_sum(runtime.zeros((2, 3), dtype=np.float32), values)
    with pytest.raises(ValueError, match="does not match"):
        accumulate_history_sums(runtime.zeros((2, 2, 2), dtype=np.float64), values)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_tracer_field_preserves_device_storage_through_views_and_reblocking(dtype):
    runtime = _cuda_runtime_or_skip()
    host = np.arange(2 * 3 * 4 * 5 * 9, dtype=dtype).reshape(2, 3, 4, 5, 9)
    names = tuple(f"tracer_{index}" for index in range(9))
    device = runtime.to_device(host)
    field = TracerField(
        names=names,
        data=device,
        units=("mol/mol",) * len(names),
        coords={},
    )

    assert runtime.is_device_array(field.block_data)
    assert runtime.is_device_array(field.block(0).block_data)
    assert runtime.is_device_array(field.tracer(8))
    assert runtime.transfer_stats.device_to_host_count == 0

    blocked = field.reblock(8)
    canonical = blocked.to_canonical()

    assert runtime.is_device_array(blocked.block_data)
    assert runtime.is_device_array(canonical)
    assert runtime.transfer_stats.device_to_host_count == 0
    actual = runtime.to_host(canonical)
    blocked_host = runtime.to_host(blocked.block_data)

    np.testing.assert_array_equal(actual, host)
    assert np.count_nonzero(blocked_host[:, -1, :, :, :, 1:]) == 0
    assert runtime.transfer_stats.host_to_device_count == 1
    assert runtime.transfer_stats.device_to_host_count == 2
