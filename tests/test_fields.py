from __future__ import annotations

import numpy as np

from wombat_transport.fields import (
    BlockedTracerField,
    TracerField,
    bottom_field3_to_top,
    canonical_time_slice,
    canonical_to_public_tracer5,
    public_surface_flux_to_transport,
    public_tracer4_to_transport,
    public_tracer5_to_canonical,
    transport_tracer_to_canonical,
    transport_tracer_to_public4,
)


def _tracer_field(ntracer: int) -> TracerField:
    data = np.arange(2 * 3 * 4 * 5 * ntracer, dtype=np.float64).reshape(2, 3, 4, 5, ntracer)
    names = tuple(f"tracer_{index}" for index in range(ntracer))
    return TracerField(names=names, data=data, units=("mol/mol",) * ntracer, coords={})


def test_single_block_field_is_zero_copy_canonical_storage():
    original = _tracer_field(7)

    blocked = BlockedTracerField.from_tracer_field(original, block_width=7)
    canonical = blocked.to_tracer_field()

    assert blocked.shape == original.shape
    assert blocked.block_count == 1
    assert blocked.block_width == 7
    assert np.shares_memory(blocked.data, original.data)
    assert np.shares_memory(canonical.data, blocked.data)
    np.testing.assert_array_equal(canonical.data, original.data)


def test_block_views_and_tracer_views_share_padded_storage():
    original = _tracer_field(9)

    blocked = BlockedTracerField.from_tracer_field(original, block_width=8)
    first, tail = tuple(blocked.iter_blocks())

    assert blocked.shape == original.shape
    assert blocked.data.shape == (2, 2, 3, 4, 5, 8)
    assert first.names == original.names[:8]
    assert tail.names == original.names[8:]
    assert np.shares_memory(first.data, blocked.data)
    assert np.shares_memory(tail.data, blocked.data)
    assert np.shares_memory(blocked.tracer(8), blocked.data)
    assert np.count_nonzero(blocked.data[:, -1, :, :, :, 1:]) == 0
    np.testing.assert_array_equal(blocked.tracer(8), original.data[..., 8])


def test_multiblock_canonical_conversion_joins_active_lanes():
    original = _tracer_field(17)
    blocked = BlockedTracerField.from_tracer_field(original, block_width=8)

    canonical = blocked.to_tracer_field()

    assert not np.shares_memory(canonical.data, blocked.data)
    np.testing.assert_array_equal(canonical.data, original.data)


def test_public_5d_tracer_roundtrip_preserves_values():
    public = np.arange(2 * 3 * 4 * 5 * 6, dtype=np.float64).reshape(2, 3, 4, 5, 6)

    canonical = public_tracer5_to_canonical(public)
    roundtrip = canonical_to_public_tracer5(canonical)

    assert canonical.shape == (3, 4, 5, 6, 2)
    np.testing.assert_array_equal(canonical[0, 0, :, :, 0], public[0, 0, -1, :, :])
    np.testing.assert_array_equal(roundtrip, public)


def test_public_4d_tracer_roundtrip_preserves_values():
    public = np.arange(2 * 4 * 5 * 6, dtype=np.float64).reshape(2, 4, 5, 6)

    transport = public_tracer4_to_transport(public)
    roundtrip = transport_tracer_to_public4(transport)

    assert transport.shape == (4, 5, 6, 2)
    np.testing.assert_array_equal(transport[0, :, :, 0], public[0, -1, :, :])
    np.testing.assert_array_equal(roundtrip, public)


def test_canonical_time_slice_and_wrap_are_inverse_for_single_time():
    transport = np.arange(4 * 5 * 6 * 2, dtype=np.float64).reshape(4, 5, 6, 2)

    canonical = transport_tracer_to_canonical(transport)

    assert canonical.shape == (1, 4, 5, 6, 2)
    np.testing.assert_array_equal(canonical_time_slice(canonical), transport)


def test_bottom_field_and_surface_flux_helpers():
    bottom = np.arange(4 * 5 * 6, dtype=np.float64).reshape(4, 5, 6)
    flux = np.arange(2 * 5 * 6, dtype=np.float64).reshape(2, 5, 6)

    top = bottom_field3_to_top(bottom)
    flux_working = public_surface_flux_to_transport(flux)

    np.testing.assert_array_equal(top[0], bottom[-1])
    assert flux_working.shape == (5, 6, 2)
    np.testing.assert_array_equal(flux_working[:, :, 0], flux[0])
