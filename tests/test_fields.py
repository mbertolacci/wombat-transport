from __future__ import annotations

import numpy as np

from wombat_transport.fields import (
    bottom_field3_to_top,
    canonical_time_slice,
    canonical_to_public_tracer5,
    public_surface_flux_to_transport,
    public_tracer4_to_transport,
    public_tracer5_to_canonical,
    transport_tracer_to_canonical,
    transport_tracer_to_public4,
)


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
