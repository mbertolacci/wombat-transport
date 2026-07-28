#define WOMBAT_MAX_LON 144
#define WOMBAT_MAX_LAT 91
#define WOMBAT_MAX_LEV 47

template <typename T>
__device__ __forceinline__ T wabs(T value) {
    return value < static_cast<T>(0) ? -value : value;
}

template <typename T>
__device__ __forceinline__ T wmin(T left, T right) {
    return left < right ? left : right;
}

template <typename T>
__device__ __forceinline__ T wmax(T left, T right) {
    return left > right ? left : right;
}

template <typename T>
__device__ __forceinline__ T wmin3(T a, T b, T c) {
    return wmin(a, wmin(b, c));
}

template <typename T>
__device__ __forceinline__ T wmax3(T a, T b, T c) {
    return wmax(a, wmax(b, c));
}

__device__ __forceinline__ int wnint(float value) {
    return __float2int_rn(value);
}

__device__ __forceinline__ int wnint(double value) {
    return __double2int_rn(value);
}

__device__ __forceinline__ int wmod(int value, int modulus) {
    const int result = value % modulus;
    return result < 0 ? result + modulus : result;
}

template <typename T>
__device__ __forceinline__ int tracer_index(
    int active_tracer,
    int lev,
    int lat,
    int lon,
    int nlev,
    int nlat,
    int nlon,
    int lane_width
) {
    const int block = active_tracer / lane_width;
    const int lane = active_tracer % lane_width;
    return (
        ((block * nlev + lev) * nlat + lat) * nlon + lon
    ) * lane_width + lane;
}

template <typename T>
__global__ void tpcore_horizontal_poles(
    T* q,
    const T* delp1,
    const T* area_1d,
    int tracer_count,
    int nlev,
    int nlat,
    int nlon,
    int lane_width
) {
    const int work = blockDim.x * blockIdx.x + threadIdx.x;
    const int work_size = nlev * tracer_count;
    if (work >= work_size) {
        return;
    }
    const int active_tracer = work % tracer_count;
    const int level = work / tracer_count;
    const int scalar_offset = level * nlat * nlon;
    T south_denom = static_cast<T>(0);
    T north_denom = static_cast<T>(0);
    T south = static_cast<T>(0);
    T north = static_cast<T>(0);
    for (int j = 0; j < 2; ++j) {
        const T area = area_1d[j];
        for (int i = 0; i < nlon; ++i) {
            const int center = scalar_offset + j * nlon + i;
            const T weight = delp1[center] * area;
            south_denom += weight;
            south += q[tracer_index<T>(
                active_tracer, level, j, i,
                nlev, nlat, nlon, lane_width
            )] * weight;
        }
    }
    for (int j = nlat - 2; j < nlat; ++j) {
        const T area = area_1d[j];
        for (int i = 0; i < nlon; ++i) {
            const int center = scalar_offset + j * nlon + i;
            const T weight = delp1[center] * area;
            north_denom += weight;
            north += q[tracer_index<T>(
                active_tracer, level, j, i,
                nlev, nlat, nlon, lane_width
            )] * weight;
        }
    }
    south /= south_denom;
    north /= north_denom;
    for (int j = 0; j < 2; ++j) {
        for (int i = 0; i < nlon; ++i) {
            q[tracer_index<T>(
                active_tracer, level, j, i,
                nlev, nlat, nlon, lane_width
            )] = south;
        }
    }
    for (int j = nlat - 2; j < nlat; ++j) {
        for (int i = 0; i < nlon; ++i) {
            q[tracer_index<T>(
                active_tracer, level, j, i,
                nlev, nlat, nlon, lane_width
            )] = north;
        }
    }
}

template <typename T>
__global__ void tpcore_horizontal_initialize(
    const T* q,
    T* dq,
    T* qqu,
    T* qqv,
    const T* delp1,
    const T* ua,
    const T* va,
    const long long* jn,
    const long long* js,
    int tracer_count,
    int nlev,
    int nlat,
    int nlon,
    int lane_width
) {
    const int work = blockDim.x * blockIdx.x + threadIdx.x;
    const int horizontal = nlat * nlon;
    const int work_size = nlev * horizontal * tracer_count;
    if (work >= work_size) {
        return;
    }
    const int active_tracer = work % tracer_count;
    const int cell = work / tracer_count;
    const int level = cell / horizontal;
    const int horizontal_cell = cell % horizontal;
    const int j = horizontal_cell / nlon;
    const int i = horizontal_cell % nlon;
    const int scalar_offset = level * horizontal;
    const int index = tracer_index<T>(
        active_tracer, level, j, i, nlev, nlat, nlon, lane_width
    );
    const T q_center = q[index];
    dq[index] = q_center * delp1[scalar_offset + horizontal_cell];
    if (j < 2 || j > nlat - 3) {
        qqu[index] = q_center;
        qqv[index] = q_center;
        return;
    }

    const T ua_value = ua[scalar_offset + horizontal_cell];
    const T va_value = va[scalar_offset + horizontal_cell];
    const int jn_level = static_cast<int>(jn[level]);
    const int js_level = static_cast<int>(js[level]);
    T qqu_value;
    if (j <= js_level || j >= jn_level) {
        const int iu0 = static_cast<int>(ua_value);
        const T ru = ua_value - static_cast<T>(iu0);
        const int iu = i - iu0;
        const int iu_mod = wmod(iu, nlon);
        if (ua_value >= static_cast<T>(0)) {
            const int im1 = wmod(iu - 1, nlon);
            const T q_i = q[tracer_index<T>(
                active_tracer, level, j, iu_mod,
                nlev, nlat, nlon, lane_width
            )];
            qqu_value = q_center + static_cast<T>(0.5) * (
                q_i + ru * (
                    q[tracer_index<T>(
                        active_tracer, level, j, im1,
                        nlev, nlat, nlon, lane_width
                    )] - q_i
                ) - q_center
            );
        } else {
            const int ip1 = wmod(iu + 1, nlon);
            const T q_i = q[tracer_index<T>(
                active_tracer, level, j, iu_mod,
                nlev, nlat, nlon, lane_width
            )];
            qqu_value = q_center + static_cast<T>(0.5) * (
                q_i + ru * (
                    q_i - q[tracer_index<T>(
                        active_tracer, level, j, ip1,
                        nlev, nlat, nlon, lane_width
                    )]
                ) - q_center
            );
        }
    } else {
        const int iu = static_cast<int>(
            static_cast<T>(i + 1) - ua_value
        ) - 1;
        const int iu_mod = wmod(iu, nlon);
        const int ip1 = wmod(iu + 1, nlon);
        qqu_value = q_center + static_cast<T>(0.5) * ua_value * (
            q[tracer_index<T>(
                active_tracer, level, j, iu_mod,
                nlev, nlat, nlon, lane_width
            )] - q[tracer_index<T>(
                active_tracer, level, j, ip1,
                nlev, nlat, nlon, lane_width
            )]
        );
    }
    qqu[index] = qqu_value;

    const int jv = static_cast<int>(
        static_cast<T>(j + 1) - va_value
    ) - 1;
    const int jvp1 = jv + 1;
    T vdiff = static_cast<T>(0);
    if (jv >= 0 && jv < nlat) {
        vdiff += q[tracer_index<T>(
            active_tracer, level, jv, i,
            nlev, nlat, nlon, lane_width
        )];
    }
    if (jvp1 >= 0 && jvp1 < nlat) {
        vdiff -= q[tracer_index<T>(
            active_tracer, level, jvp1, i,
            nlev, nlat, nlon, lane_width
        )];
    }
    qqv[index] = q_center + static_cast<T>(0.5) * va_value * vdiff;
}

template <typename T>
__global__ void tpcore_horizontal_zonal_warp(
    T* q,
    T* dq,
    T* qqu,
    T* qqv,
    const T* pu,
    const T* xmass,
    const T* cx,
    const T* ua,
    const T* va,
    const long long* jn,
    const long long* js,
    int tracer_count,
    int nlev,
    int nlat,
    int nlon,
    int lane_width
) {
    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int work = blockIdx.x * warps_per_block + warp_in_block;
    const int work_size = nlev * tracer_count;
    if (work >= work_size || nlon > WOMBAT_MAX_LON || nlat > WOMBAT_MAX_LAT) {
        return;
    }
    const int active_tracer = work % tracer_count;
    const int level = work / tracer_count;
    const int horizontal = nlat * nlon;
    const int scalar_offset = level * horizontal;
    const int j1p = 2;
    const int j2p = nlat - 3;
    const int jn_level = static_cast<int>(jn[level]);
    const int js_level = static_cast<int>(js[level]);
    const T r13 = static_cast<T>(1) / static_cast<T>(3);
    const T r23 = static_cast<T>(2) / static_cast<T>(3);
    const T r24 = static_cast<T>(1) / static_cast<T>(24);
    extern __shared__ T shared[];
    T* warp_shared = shared + warp_in_block * 4 * WOMBAT_MAX_LON;
    T* w0 = warp_shared;
    T* w1 = w0 + WOMBAT_MAX_LON;
    T* w2 = w1 + WOMBAT_MAX_LON;
    T* fluxes = w2 + WOMBAT_MAX_LON;

    T south = static_cast<T>(0);
    T north = static_cast<T>(0);

    const int interior_cells = (j2p - j1p + 1) * nlon;
    for (int cell = lane; cell < interior_cells; cell += 32) {
        const int j = j1p + cell / nlon;
        const int i = cell % nlon;
        const T ua_value = ua[scalar_offset + j * nlon + i];
        const int iu0 = wnint(ua_value);
        const T ru = static_cast<T>(iu0) - ua_value;
        const int iu = i - iu0;
        const int im1 = wmod(iu - 1, nlon);
        const int iu_mod = wmod(iu, nlon);
        const int ip1 = wmod(iu + 1, nlon);
        const T q_i = qqv[tracer_index<T>(
            active_tracer, level, j, iu_mod,
            nlev, nlat, nlon, lane_width
        )];
        const T q_ip1 = qqv[tracer_index<T>(
            active_tracer, level, j, ip1,
            nlev, nlat, nlon, lane_width
        )];
        const T q_im1 = qqv[tracer_index<T>(
            active_tracer, level, j, im1,
            nlev, nlat, nlon, lane_width
        )];
        const T a1 = static_cast<T>(0.5) * (q_ip1 + q_im1) - q_i;
        const T b1 = static_cast<T>(0.5) * (q_ip1 - q_im1);
        const int index = tracer_index<T>(
            active_tracer, level, j, i, nlev, nlat, nlon, lane_width
        );
        q[index] += ru * (
            a1 * ru + b1
        ) + q_i - qqv[index];
    }

    for (int cell = lane; cell < interior_cells; cell += 32) {
        const int j = j1p + cell / nlon;
        const int i = cell % nlon;
        const T va_value = va[scalar_offset + j * nlon + i];
        const int jv0 = wnint(va_value);
        const T rv = static_cast<T>(jv0) - va_value;
        const int jv = j - jv0;
        const int jm1 = jv - 1;
        const int jp1 = jv + 1;
        const T q_j = jv >= 0 && jv < nlat
            ? qqu[tracer_index<T>(
                active_tracer, level, jv, i,
                nlev, nlat, nlon, lane_width
            )] : static_cast<T>(0);
        const T q_jp1 = jp1 >= 0 && jp1 < nlat
            ? qqu[tracer_index<T>(
                active_tracer, level, jp1, i,
                nlev, nlat, nlon, lane_width
            )] : static_cast<T>(0);
        const T q_jm1 = jm1 >= 0 && jm1 < nlat
            ? qqu[tracer_index<T>(
                active_tracer, level, jm1, i,
                nlev, nlat, nlon, lane_width
            )] : static_cast<T>(0);
        const T a1 = static_cast<T>(0.5) * (q_jp1 + q_jm1) - q_j;
        const T b1 = static_cast<T>(0.5) * (q_jp1 - q_jm1);
        const int index = tracer_index<T>(
            active_tracer, level, j, i, nlev, nlat, nlon, lane_width
        );
        q[index] += rv * (
            a1 * rv + b1
        ) + q_j - qqu[index];
    }
    __syncwarp();
    T sumsp = static_cast<T>(0);
    T sumnp = static_cast<T>(0);
    for (int i = lane; i < nlon; i += 32) {
        int j = 1;
        T va_value = va[scalar_offset + j * nlon + i];
        int jv0 = wnint(va_value);
        T rv = static_cast<T>(jv0) - va_value;
        int jv = j - jv0;
        int jm1 = jv - 1;
        int jp1 = jv + 1;
        T q_j = jv >= 0 && jv < nlat
            ? qqu[tracer_index<T>(
                active_tracer, level, jv, i,
                nlev, nlat, nlon, lane_width
            )] : static_cast<T>(0);
        T q_jp1 = jp1 >= 0 && jp1 < nlat
            ? qqu[tracer_index<T>(
                active_tracer, level, jp1, i,
                nlev, nlat, nlon, lane_width
            )] : static_cast<T>(0);
        T q_jm1 = jm1 >= 0 && jm1 < nlat
            ? qqu[tracer_index<T>(
                active_tracer, level, jm1, i,
                nlev, nlat, nlon, lane_width
            )] : static_cast<T>(0);
        T a1 = static_cast<T>(0.5) * (q_jp1 + q_jm1) - q_j;
        T b1 = static_cast<T>(0.5) * (q_jp1 - q_jm1);
        const int south_index = tracer_index<T>(
            active_tracer, level, j, i, nlev, nlat, nlon, lane_width
        );
        w0[i] = rv * (a1 * rv + b1) + q_j - qqu[south_index];

        j = nlat - 2;
        va_value = va[scalar_offset + j * nlon + i];
        jv0 = wnint(va_value);
        rv = static_cast<T>(jv0) - va_value;
        jv = j - jv0;
        jm1 = jv - 1;
        jp1 = jv + 1;
        q_j = jv >= 0 && jv < nlat
            ? qqu[tracer_index<T>(
                active_tracer, level, jv, i,
                nlev, nlat, nlon, lane_width
            )] : static_cast<T>(0);
        q_jp1 = jp1 >= 0 && jp1 < nlat
            ? qqu[tracer_index<T>(
                active_tracer, level, jp1, i,
                nlev, nlat, nlon, lane_width
            )] : static_cast<T>(0);
        q_jm1 = jm1 >= 0 && jm1 < nlat
            ? qqu[tracer_index<T>(
                active_tracer, level, jm1, i,
                nlev, nlat, nlon, lane_width
            )] : static_cast<T>(0);
        a1 = static_cast<T>(0.5) * (q_jp1 + q_jm1) - q_j;
        b1 = static_cast<T>(0.5) * (q_jp1 - q_jm1);
        const int north_index = tracer_index<T>(
            active_tracer, level, j, i, nlev, nlat, nlon, lane_width
        );
        w1[i] = rv * (a1 * rv + b1) + q_j - qqu[north_index];
    }
    __syncwarp();
    if (lane == 0) {
        south = static_cast<T>(0);
        north = static_cast<T>(0);
        for (int i = 0; i < nlon; ++i) {
            south += w0[i];
            north += w1[i];
        }
        w2[0] = south / static_cast<T>(nlon);
        w2[1] = north / static_cast<T>(nlon);
    }
    __syncwarp();
    south = w2[0];
    north = w2[1];
    for (int i = lane; i < nlon; i += 32) {
        q[tracer_index<T>(
            active_tracer, level, 0, i, nlev, nlat, nlon, lane_width
        )] += south;
        q[tracer_index<T>(
            active_tracer, level, 1, i, nlev, nlat, nlon, lane_width
        )] += south;
        q[tracer_index<T>(
            active_tracer, level, nlat - 2, i,
            nlev, nlat, nlon, lane_width
        )] += north;
        q[tracer_index<T>(
            active_tracer, level, nlat - 1, i,
            nlev, nlat, nlon, lane_width
        )] += north;
    }
    __syncwarp();

    const int jvan = wmax(1, nlat / 18);
    for (int j = j1p; j <= j2p; ++j) {
        if (j > j1p && j < j2p) {
            for (int i = lane; i < nlon; i += 32) {
                const int im1 = wmod(i - 1, nlon);
                const int ip1 = wmod(i + 1, nlon);
                const int im2 = wmod(i - 2, nlon);
                const int ip2 = wmod(i + 2, nlon);
                const T q_im1 = qqv[tracer_index<T>(
                    active_tracer, level, j, im1,
                    nlev, nlat, nlon, lane_width
                )];
                const T q_i = qqv[tracer_index<T>(
                    active_tracer, level, j, i,
                    nlev, nlat, nlon, lane_width
                )];
                const T q_ip1 = qqv[tracer_index<T>(
                    active_tracer, level, j, ip1,
                    nlev, nlat, nlon, lane_width
                )];
                const T tmp = (
                    static_cast<T>(8) * (q_ip1 - q_im1) +
                    qqv[tracer_index<T>(
                        active_tracer, level, j, im2,
                        nlev, nlat, nlon, lane_width
                    )] - qqv[tracer_index<T>(
                        active_tracer, level, j, ip2,
                        nlev, nlat, nlon, lane_width
                    )]
                ) * r24;
                const T pmax = wmax3(q_im1, q_i, q_ip1) - q_i;
                const T pmin = q_i - wmin3(q_im1, q_i, q_ip1);
                const T bounded = wmin(wabs(tmp), wmin(pmin, pmax));
                w0[i] = tmp >= static_cast<T>(0) ? bounded : -bounded;
            }
        }
        __syncwarp();

        const bool local_courant = j > js_level && j < jn_level;
        const bool edge_row = j == j1p || j == j2p;
        const bool near_edge_row = j <= j1p + jvan || j >= j2p - jvan;
        if (local_courant && !edge_row && !near_edge_row) {
            for (int i = lane; i < nlon; i += 32) {
                const int im1 = wmod(i - 1, nlon);
                w1[i] = static_cast<T>(0.5) * (
                    qqv[tracer_index<T>(
                        active_tracer, level, j, im1,
                        nlev, nlat, nlon, lane_width
                    )] + qqv[tracer_index<T>(
                        active_tracer, level, j, i,
                        nlev, nlat, nlon, lane_width
                    )]
                ) + (w0[im1] - w0[i]) * r13;
                w2[im1] = w1[i];
            }
            __syncwarp();
            for (int i = lane; i < nlon; i += 32) {
                const T q_value = qqv[tracer_index<T>(
                    active_tracer, level, j, i,
                    nlev, nlat, nlon, lane_width
                )];
                T a6 = static_cast<T>(3) * (
                    q_value + q_value - (w1[i] + w2[i])
                );
                if (w0[i] == static_cast<T>(0)) {
                    a6 = static_cast<T>(0);
                    w1[i] = q_value;
                    w2[i] = q_value;
                } else {
                    const T da1 = w2[i] - w1[i];
                    const T da2 = da1 * da1;
                    const T a6da = a6 * da1;
                    if (a6da < -da2) {
                        a6 = static_cast<T>(3) * (w1[i] - q_value);
                        w2[i] = w1[i] - a6;
                    } else if (a6da > da2) {
                        a6 = static_cast<T>(3) * (w2[i] - q_value);
                        w1[i] = w2[i] - a6;
                    }
                }
                w0[i] = a6;
            }
            __syncwarp();
        }
        for (int i = lane; i < nlon; i += 32) {
            const T c = cx[scalar_offset + j * nlon + i];
            T flux;
            if (local_courant) {
                if (edge_row) {
                    const int iu = wmod(
                        static_cast<int>(static_cast<T>(i + 1) - c) - 1,
                        nlon
                    );
                    flux = qqv[tracer_index<T>(
                        active_tracer, level, j, iu,
                        nlev, nlat, nlon, lane_width
                    )];
                } else if (near_edge_row) {
                    const int iu = wmod(
                        static_cast<int>(static_cast<T>(i + 1) - c) - 1,
                        nlon
                    );
                    const T sign_value = c >= static_cast<T>(0)
                        ? static_cast<T>(1) : static_cast<T>(-1);
                    flux = qqv[tracer_index<T>(
                        active_tracer, level, j, iu,
                        nlev, nlat, nlon, lane_width
                    )] + w0[iu] * (sign_value - c);
                } else if (c > static_cast<T>(0)) {
                    const int im1 = wmod(i - 1, nlon);
                    flux = w2[im1] + static_cast<T>(0.5) * c * (
                        w1[im1] - w2[im1] +
                        w0[im1] * (
                            static_cast<T>(1) - r23 * c
                        )
                    );
                } else {
                    flux = w1[i] - static_cast<T>(0.5) * c * (
                        w2[i] - w1[i] +
                        w0[i] * (
                            static_cast<T>(1) + r23 * c
                        )
                    );
                }
                flux *= xmass[scalar_offset + j * nlon + i];
            } else {
                const int ic = static_cast<int>(c);
                const int isav = i - ic;
                const int iu = wmod(
                    static_cast<int>(static_cast<T>(i + 1) - c) - 1,
                    nlon
                );
                const T rc = c - static_cast<T>(ic);
                const T sign_value = rc >= static_cast<T>(0)
                    ? static_cast<T>(1) : static_cast<T>(-1);
                T value;
                if (edge_row) {
                    value = rc * qqv[tracer_index<T>(
                        active_tracer, level, j, iu,
                        nlev, nlat, nlon, lane_width
                    )];
                } else {
                    value = rc * (
                        qqv[tracer_index<T>(
                            active_tracer, level, j, iu,
                            nlev, nlat, nlon, lane_width
                        )] + w0[iu] * (sign_value - rc)
                    );
                }
                if (c > static_cast<T>(1)) {
                    for (int ix = isav; ix < i; ++ix) {
                        value += qqv[tracer_index<T>(
                            active_tracer, level, j, wmod(ix, nlon),
                            nlev, nlat, nlon, lane_width
                        )];
                    }
                } else if (c < static_cast<T>(-1)) {
                    for (int ix = i; ix < isav; ++ix) {
                        value -= qqv[tracer_index<T>(
                            active_tracer, level, j, wmod(ix, nlon),
                            nlev, nlat, nlon, lane_width
                        )];
                    }
                }
                flux = pu[scalar_offset + j * nlon + i] * value;
            }
            fluxes[i] = flux;
        }
        __syncwarp();
        for (int i = lane; i < nlon; i += 32) {
            const int ip1 = i + 1 == nlon ? 0 : i + 1;
            dq[tracer_index<T>(
                active_tracer, level, j, i,
                nlev, nlat, nlon, lane_width
            )] += fluxes[i] - fluxes[ip1];
        }
        __syncwarp();
    }

}

template <typename T>
__global__ void tpcore_horizontal_meridional(
    T* __restrict__ dq,
    const T* __restrict__ qqu,
    T* __restrict__ qqv,
    const T* __restrict__ ymass,
    const T* __restrict__ cy,
    const T* __restrict__ geofac,
    int tracer_count,
    int nlev,
    int nlat,
    int nlon,
    int lane_width
) {
    const int work = blockDim.x * blockIdx.x + threadIdx.x;
    const int work_size = nlev * nlon * tracer_count;
    if (work >= work_size || nlat > WOMBAT_MAX_LAT) {
        return;
    }
    const int active_tracer = work % tracer_count;
    const int column = work / tracer_count;
    const int i = column % nlon;
    const int level = column / nlon;
    const int horizontal = nlat * nlon;
    const int scalar_offset = level * horizontal;
    const int j1p = 2;
    const int j2p = nlat - 3;
    const T r13 = static_cast<T>(1) / static_cast<T>(3);
    const T r23 = static_cast<T>(2) / static_cast<T>(3);
    const T r24 = static_cast<T>(1) / static_cast<T>(24);
    T w0[WOMBAT_MAX_LAT];
    T w1[WOMBAT_MAX_LAT];
    T w2[WOMBAT_MAX_LAT];

    w0[0] = static_cast<T>(0);
    w0[nlat - 1] = static_cast<T>(0);
    for (int j = 1; j < nlat - 1; ++j) {
        const T qjm2 = j < 2 ? static_cast<T>(0) :
            qqu[tracer_index<T>(
                active_tracer, level, j - 2, i,
                nlev, nlat, nlon, lane_width
            )];
        const T qjm1 = qqu[tracer_index<T>(
            active_tracer, level, j - 1, i,
            nlev, nlat, nlon, lane_width
        )];
        const T qj = qqu[tracer_index<T>(
            active_tracer, level, j, i,
            nlev, nlat, nlon, lane_width
        )];
        const T qjp1 = qqu[tracer_index<T>(
            active_tracer, level, j + 1, i,
            nlev, nlat, nlon, lane_width
        )];
        const T qjp2 = j + 2 >= nlat ? static_cast<T>(0) :
            qqu[tracer_index<T>(
                active_tracer, level, j + 2, i,
                nlev, nlat, nlon, lane_width
            )];
        const T tmp = (
            static_cast<T>(8) * (qjp1 - qjm1) + qjm2 - qjp2
        ) * r24;
        const T pmax = wmax3(qjm1, qj, qjp1) - qj;
        const T pmin = qj - wmin3(qjm1, qj, qjp1);
        const T bounded = wmin(wabs(tmp), wmin(pmin, pmax));
        w0[j] = tmp >= static_cast<T>(0) ? bounded : -bounded;
    }
    for (int j = 1; j < nlat; ++j) {
        w1[j] = static_cast<T>(0.5) * (
            qqu[tracer_index<T>(
                active_tracer, level, j - 1, i,
                nlev, nlat, nlon, lane_width
            )] + qqu[tracer_index<T>(
                active_tracer, level, j, i,
                nlev, nlat, nlon, lane_width
            )]
        ) + (w0[j - 1] - w0[j]) * r13;
        w2[j - 1] = w1[j];
    }
    for (int j = 1; j < nlat - 1; ++j) {
        const T qj = qqu[tracer_index<T>(
            active_tracer, level, j, i,
            nlev, nlat, nlon, lane_width
        )];
        const T slope = w0[j];
        w0[j] = static_cast<T>(3) * (
            qj + qj - (w1[j] + w2[j])
        );
        if (slope == static_cast<T>(0)) {
            w0[j] = static_cast<T>(0);
            w1[j] = qj;
            w2[j] = qj;
        } else {
            const T da1 = w2[j] - w1[j];
            const T da2 = da1 * da1;
            const T a6da = w0[j] * da1;
            if (a6da < -da2) {
                w0[j] = static_cast<T>(3) * (w1[j] - qj);
                w2[j] = w1[j] - w0[j];
            } else if (a6da > da2) {
                w0[j] = static_cast<T>(3) * (w2[j] - qj);
                w1[j] = w2[j] - w0[j];
            }
        }
    }
    for (int j = j1p; j < j2p + 2; ++j) {
        const int jm1 = j - 1;
        const T c = cy[scalar_offset + j * nlon + i];
        T flux;
        if (c > static_cast<T>(0)) {
            flux = w2[jm1] + static_cast<T>(0.5) * c * (
                w1[jm1] - w2[jm1] +
                w0[jm1] * (static_cast<T>(1) - r23 * c)
            );
        } else {
            flux = w1[j] - static_cast<T>(0.5) * c * (
                w2[j] - w1[j] +
                w0[j] * (static_cast<T>(1) + r23 * c)
            );
        }
        qqv[tracer_index<T>(
            active_tracer, level, j, i,
            nlev, nlat, nlon, lane_width
        )] = flux;
    }
    int index = tracer_index<T>(
        active_tracer, level, j1p, i,
        nlev, nlat, nlon, lane_width
    );
    qqv[index] *= ymass[scalar_offset + j1p * nlon + i];
    for (int j = j1p; j <= j2p; ++j) {
        const int next_index = tracer_index<T>(
            active_tracer, level, j + 1, i,
            nlev, nlat, nlon, lane_width
        );
        qqv[next_index] *= ymass[scalar_offset + (j + 1) * nlon + i];
        dq[tracer_index<T>(
            active_tracer, level, j, i,
            nlev, nlat, nlon, lane_width
        )] += (qqv[index] - qqv[next_index]) * geofac[j];
        index = next_index;
    }
}

template <typename T>
__global__ void tpcore_horizontal_finalize_poles(
    T* dq,
    const T* qqv,
    T geofac_pc,
    int tracer_count,
    int nlev,
    int nlat,
    int nlon,
    int lane_width
) {
    const int work = blockDim.x * blockIdx.x + threadIdx.x;
    const int work_size = nlev * tracer_count;
    if (work >= work_size) {
        return;
    }
    const int active_tracer = work % tracer_count;
    const int level = work / tracer_count;
    const int j1p = 2;
    const int j2p = nlat - 3;
    T sumsp = static_cast<T>(0);
    T sumnp = static_cast<T>(0);
    for (int i = 0; i < nlon; ++i) {
        sumsp += qqv[tracer_index<T>(
            active_tracer, level, j1p, i,
            nlev, nlat, nlon, lane_width
        )];
        sumnp += qqv[tracer_index<T>(
            active_tracer, level, j2p + 1, i,
            nlev, nlat, nlon, lane_width
        )];
    }
    const T dq_sp = dq[tracer_index<T>(
        active_tracer, level, 0, 0,
        nlev, nlat, nlon, lane_width
    )] - sumsp / static_cast<T>(nlon) * geofac_pc;
    const T dq_np = dq[tracer_index<T>(
        active_tracer, level, nlat - 1, 0,
        nlev, nlat, nlon, lane_width
    )] + sumnp / static_cast<T>(nlon) * geofac_pc;
    for (int i = 0; i < nlon; ++i) {
        dq[tracer_index<T>(
            active_tracer, level, 0, i,
            nlev, nlat, nlon, lane_width
        )] = dq_sp;
        dq[tracer_index<T>(
            active_tracer, level, 1, i,
            nlev, nlat, nlon, lane_width
        )] = dq_sp;
        dq[tracer_index<T>(
            active_tracer, level, nlat - 2, i,
            nlev, nlat, nlon, lane_width
        )] = dq_np;
        dq[tracer_index<T>(
            active_tracer, level, nlat - 1, i,
            nlev, nlat, nlon, lane_width
        )] = dq_np;
    }
}

template <typename T>
__global__ void tpcore_prepare_vertical_coefficients(
    const T* delp1,
    T* slope_c0,
    T* slope_c1,
    T* slope_c2,
    T* interface_c2,
    T* interface_a1,
    T* interface_a2,
    T* top_fac2,
    T* top_ratio,
    T* bottom_ratio,
    int nlev,
    int nlat,
    int nlon
) {
    const int work = blockDim.x * blockIdx.x + threadIdx.x;
    const int ncol = nlat * nlon;
    const int work_size = nlev * ncol;
    if (work >= work_size) {
        return;
    }
    const int k = work / ncol;
    const int col = work % ncol;

    if (k >= 1 && k < nlev - 1) {
        const T dlp_km1 = delp1[(k - 1) * ncol + col];
        const T dlp_k = delp1[k * ncol + col];
        const T dlp_kp1 = delp1[(k + 1) * ncol + col];
        slope_c0[work] = dlp_k / (dlp_km1 + dlp_k + dlp_kp1);
        slope_c1[work] = (
            dlp_km1 + static_cast<T>(0.5) * dlp_k
        ) / (dlp_kp1 + dlp_k);
        slope_c2[work] = (
            dlp_kp1 + static_cast<T>(0.5) * dlp_k
        ) / (dlp_km1 + dlp_k);
    }

    if (k >= 2 && k < nlev - 1) {
        const T dlp_km2 = delp1[(k - 2) * ncol + col];
        const T dlp_km1 = delp1[(k - 1) * ncol + col];
        const T dlp_k = delp1[k * ncol + col];
        const T dlp_kp1 = delp1[(k + 1) * ncol + col];
        interface_c2[work] = static_cast<T>(2) /
            (dlp_km2 + dlp_km1 + dlp_k + dlp_kp1);
        interface_a1[work] = (dlp_km2 + dlp_km1) /
            (static_cast<T>(2) * dlp_km1 + dlp_k);
        interface_a2[work] = (dlp_k + dlp_kp1) /
            (static_cast<T>(2) * dlp_k + dlp_km1);
    }

    if (k == 0) {
        const T dlp0 = delp1[col];
        const T dlp1_value = delp1[ncol + col];
        const T dlp2_value = delp1[2 * ncol + col];
        top_fac2[col] = (dlp1_value + dlp2_value) *
            (dlp0 + dlp1_value + dlp2_value);
        top_ratio[col] = (dlp1_value + dlp2_value) /
            (dlp0 + dlp1_value);

        const T dlp_last = delp1[(nlev - 1) * ncol + col];
        const T dlp_prev = delp1[(nlev - 2) * ncol + col];
        bottom_ratio[col] = (dlp_last * dlp_last) /
            ((dlp_last + dlp_prev) * (
                static_cast<T>(2) * dlp_last + dlp_prev
            ));
    }
}

template <typename T, bool UsePreparedCoefficients>
__global__ void tpcore_vertical(
    const T* q,
    T* dq,
    const T* delp1,
    const T* delp2,
    const T* wz,
    const T* normalized_vertical_courant,
    const T* slope_c0,
    const T* slope_c1,
    const T* slope_c2,
    const T* interface_c2,
    const T* interface_a1,
    const T* interface_a2,
    const T* top_fac2,
    const T* top_ratio,
    const T* bottom_ratio,
    int fill,
    int finalize_output,
    int tracer_count,
    int nlev,
    int nlat,
    int nlon,
    int lane_width
) {
    const int work = blockDim.x * blockIdx.x + threadIdx.x;
    const int ncol = nlat * nlon;
    const int work_size = ncol * tracer_count;
    if (work >= work_size || nlev > WOMBAT_MAX_LEV) {
        return;
    }
    const int active_tracer = work % tracer_count;
    const int col = work / tracer_count;
    const int lat = col / nlon;
    const int lon = col % nlon;
    if (lat == 1 || lat == nlat - 2) {
        return;
    }
    // Base-plus-stride indexing wins for float32 but raises float64 register use.
#if WOMBAT_TPCORE_HOIST_VERTICAL_INDEX
    const int tracer_block = active_tracer / lane_width;
    const int tracer_lane = active_tracer % lane_width;
    const int tracer_level_stride = ncol * lane_width;
    const int tracer_row_stride = nlon * lane_width;
    const int tracer_base = (
        tracer_block * nlev * ncol + col
    ) * lane_width + tracer_lane;
#define WOMBAT_VERTICAL_INDEX(level, target_lat) ( \
    tracer_base + (level) * tracer_level_stride + \
    ((target_lat) - lat) * tracer_row_stride \
)
#define WOMBAT_VERTICAL_LEVEL_OFFSET(index, level, offset) ( \
    (index) + (offset) * tracer_level_stride \
)
#define WOMBAT_VERTICAL_ROW_OFFSET(index, level, target_lat) ( \
    (index) + ((target_lat) - lat) * tracer_row_stride \
)
#else
#define WOMBAT_VERTICAL_INDEX(level, target_lat) tracer_index<T>( \
    active_tracer, level, target_lat, lon, \
    nlev, nlat, nlon, lane_width \
)
#define WOMBAT_VERTICAL_LEVEL_OFFSET(index, level, offset) \
    WOMBAT_VERTICAL_INDEX((level) + (offset), lat)
#define WOMBAT_VERTICAL_ROW_OFFSET(index, level, target_lat) \
    WOMBAT_VERTICAL_INDEX(level, target_lat)
#endif
    const T r13 = static_cast<T>(1) / static_cast<T>(3);
    const T r23 = static_cast<T>(2) / static_cast<T>(3);
    T dpi[WOMBAT_MAX_LEV];
    T dc[WOMBAT_MAX_LEV];
    T al[WOMBAT_MAX_LEV];
    T ar[WOMBAT_MAX_LEV];
    T a6_top;
    T a6_second;
    T a6_penultimate;
    T a6_bottom;
    T dca_top;
    T dca_bottom;

    for (int k = 0; k < nlev - 1; ++k) {
        const int index = WOMBAT_VERTICAL_INDEX(k, lat);
        dpi[k] = q[WOMBAT_VERTICAL_LEVEL_OFFSET(index, k, 1)] - q[index];
    }
    for (int k = 1; k < nlev - 1; ++k) {
        T c0;
        T c1;
        T c2;
        if (UsePreparedCoefficients) {
            const int pressure_index = k * ncol + col;
            c0 = slope_c0[pressure_index];
            c1 = slope_c1[pressure_index];
            c2 = slope_c2[pressure_index];
        } else {
            const T dlp_km1 = delp1[(k - 1) * ncol + col];
            const T dlp_k = delp1[k * ncol + col];
            const T dlp_kp1 = delp1[(k + 1) * ncol + col];
            c0 = dlp_k / (dlp_km1 + dlp_k + dlp_kp1);
            c1 = (dlp_km1 + static_cast<T>(0.5) * dlp_k) /
                (dlp_kp1 + dlp_k);
            c2 = (dlp_kp1 + static_cast<T>(0.5) * dlp_k) /
                (dlp_km1 + dlp_k);
        }
        const T tmp = c0 * (c1 * dpi[k] + c2 * dpi[k - 1]);
        const int index = WOMBAT_VERTICAL_INDEX(k, lat);
        const T q_center = q[index];
        const T q_prev = q[WOMBAT_VERTICAL_LEVEL_OFFSET(index, k, -1)];
        const T q_next = q[WOMBAT_VERTICAL_LEVEL_OFFSET(index, k, 1)];
        const T qmax = wmax3(q_prev, q_center, q_next) - q_center;
        const T qmin = q_center - wmin3(q_prev, q_center, q_next);
        const T bounded = wmin(wabs(tmp), wmin(qmax, qmin));
        dc[k] = tmp >= static_cast<T>(0) ? bounded : -bounded;
    }

    const T dlp0 = delp1[col];
    const T dlp1_value = delp1[ncol + col];
    T fac2;
    T top_ratio_value;
    if (UsePreparedCoefficients) {
        fac2 = top_fac2[col];
        top_ratio_value = top_ratio[col];
    } else {
        const T dlp2_value = delp1[2 * ncol + col];
        fac2 = (dlp1_value + dlp2_value) *
            (dlp0 + dlp1_value + dlp2_value);
        top_ratio_value = (dlp1_value + dlp2_value) /
            (dlp0 + dlp1_value);
    }
    const T fac1 = dpi[1] - dpi[0] * top_ratio_value;
    const T aa = static_cast<T>(3) * fac1 / fac2;
    T bb = static_cast<T>(2) * dpi[0] / (dlp0 + dlp1_value);
    bb -= r23 * aa * (static_cast<T>(2) * dlp0 + dlp1_value);
    al[0] = q[WOMBAT_VERTICAL_INDEX(0, lat)] -
        dlp0 * (r13 * aa * dlp0 + static_cast<T>(0.5) * bb);
    al[1] = dlp0 * (aa * dlp0 + bb) + al[0];
    const T q_top = q[WOMBAT_VERTICAL_INDEX(0, lat)];
    if (q_top * al[0] <= static_cast<T>(0)) {
        al[0] = static_cast<T>(0);
        dca_top = static_cast<T>(0);
    } else {
        dca_top = q_top - al[0];
    }

    T bottom_ratio_value;
    if (UsePreparedCoefficients) {
        bottom_ratio_value = bottom_ratio[col];
    } else {
        const T dlp_last = delp1[(nlev - 1) * ncol + col];
        const T dlp_prev = delp1[(nlev - 2) * ncol + col];
        bottom_ratio_value = (dlp_last * dlp_last) /
            ((dlp_last + dlp_prev) * (
                static_cast<T>(2) * dlp_last + dlp_prev
            ));
    }
    const T fac1b = dpi[nlev - 2] * bottom_ratio_value;
    const T q_bottom = q[WOMBAT_VERTICAL_INDEX(nlev - 1, lat)];
    ar[nlev - 1] = q_bottom + fac1b;
    al[nlev - 1] = q_bottom - (fac1b + fac1b);
    if (q_bottom * ar[nlev - 1] <= static_cast<T>(0)) {
        ar[nlev - 1] = static_cast<T>(0);
    }
    dca_bottom = ar[nlev - 1] - q_bottom;

    for (int k = 2; k < nlev - 1; ++k) {
        const T dlp_km1 = delp1[(k - 1) * ncol + col];
        const T dlp_k = delp1[k * ncol + col];
        T c2;
        T a1_value;
        T a2_value;
        if (UsePreparedCoefficients) {
            const int pressure_index = k * ncol + col;
            c2 = interface_c2[pressure_index];
            a1_value = interface_a1[pressure_index];
            a2_value = interface_a2[pressure_index];
        } else {
            const T dlp_km2 = delp1[(k - 2) * ncol + col];
            const T dlp_kp1 = delp1[(k + 1) * ncol + col];
            c2 = static_cast<T>(2) /
                (dlp_km2 + dlp_km1 + dlp_k + dlp_kp1);
            a1_value = (dlp_km2 + dlp_km1) /
                (static_cast<T>(2) * dlp_km1 + dlp_k);
            a2_value = (dlp_k + dlp_kp1) /
                (static_cast<T>(2) * dlp_k + dlp_km1);
        }
        const T c1_value = dpi[k - 1] * dlp_km1 /
            (dlp_km1 + dlp_k);
        al[k] = q[WOMBAT_VERTICAL_INDEX(k - 1, lat)] +
            c1_value + c2 * (
            dlp_k * (
                c1_value * (a1_value - a2_value) +
                a2_value * dc[k - 1]
            ) - dlp_km1 * a1_value * dc[k]
        );
    }
    for (int k = 0; k < nlev - 1; ++k) {
        ar[k] = al[k + 1];
    }

    for (int endpoint = 0; endpoint < 2; ++endpoint) {
        const int k = endpoint == 0 ? 0 : nlev - 1;
        const T qa = q[WOMBAT_VERTICAL_INDEX(k, lat)];
        T a6_value = static_cast<T>(3) * (
            qa + qa - (al[k] + ar[k])
        );
        const T dca = endpoint == 0 ? dca_top : dca_bottom;
        if (dca == static_cast<T>(0)) {
            a6_value = static_cast<T>(0);
            al[k] = qa;
            ar[k] = qa;
        } else {
            const T da1 = ar[k] - al[k];
            const T da2 = da1 * da1;
            const T a6da = a6_value * da1;
            if (a6da < -da2) {
                a6_value = static_cast<T>(3) * (al[k] - qa);
                ar[k] = al[k] - a6_value;
            } else if (a6da > da2) {
                a6_value = static_cast<T>(3) * (ar[k] - qa);
                al[k] = ar[k] - a6_value;
            }
        }
        if (endpoint == 0) {
            a6_top = a6_value;
        } else {
            a6_bottom = a6_value;
        }
    }
    for (int endpoint = 0; endpoint < 2; ++endpoint) {
        const int k = endpoint == 0 ? 1 : nlev - 2;
        const T qa = q[WOMBAT_VERTICAL_INDEX(k, lat)];
        T a6_value = static_cast<T>(3) * (
            qa + qa - (al[k] + ar[k])
        );
        if (dc[k] == static_cast<T>(0)) {
            a6_value = static_cast<T>(0);
            al[k] = qa;
            ar[k] = qa;
        } else {
            const T da1 = ar[k] - al[k];
            const T da2 = da1 * da1;
            const T a6da = a6_value * da1;
            if (a6da < -da2) {
                a6_value = static_cast<T>(3) * (al[k] - qa);
                ar[k] = al[k] - a6_value;
            } else if (a6da > da2) {
                a6_value = static_cast<T>(3) * (ar[k] - qa);
                al[k] = ar[k] - a6_value;
            }
        }
        if (endpoint == 0) {
            a6_second = a6_value;
        } else {
            a6_penultimate = a6_value;
        }
    }
    for (int k = 1; k < nlev - 1; ++k) {
        dc[k] = dpi[k] - dpi[k - 1];
    }
    for (int k = 2; k < nlev - 2; ++k) {
        const T qq = q[WOMBAT_VERTICAL_INDEX(k, lat)];
        T qmp = qq + static_cast<T>(2) * dpi[k - 1];
        T lac = qq + static_cast<T>(1.5) * dc[k - 1] +
            static_cast<T>(0.5) * dpi[k - 1];
        T qmin = wmin3(qq, qmp, lac);
        T qmax = wmax3(qq, qmp, lac);
        ar[k] = wmax(qmin, wmin(qmax, ar[k]));
        qmp = qq - static_cast<T>(2) * dpi[k];
        lac = qq + static_cast<T>(1.5) * dc[k + 1] -
            static_cast<T>(0.5) * dpi[k];
        qmin = wmin3(qq, qmp, lac);
        qmax = wmax3(qq, qmp, lac);
        al[k] = wmax(qmin, wmin(qmax, al[k]));
    }

    T courant = normalized_vertical_courant[col];
    T previous_flux;
    if (courant > static_cast<T>(0)) {
        const T value = ar[0] + static_cast<T>(0.5) * courant * (
            al[0] - ar[0] +
            a6_top * (static_cast<T>(1) - r23 * courant)
        );
        previous_flux = wz[col] * value;
    } else {
        const T value = al[1] + static_cast<T>(0.5) * courant * (
            al[1] - ar[1] -
            a6_second * (static_cast<T>(1) + r23 * courant)
        );
        previous_flux = wz[col] * value;
    }
    dq[WOMBAT_VERTICAL_INDEX(0, lat)] -= previous_flux;
    for (int k = 1; k < nlev - 1; ++k) {
        courant = normalized_vertical_courant[k * ncol + col];
        T value;
        if (courant > static_cast<T>(0)) {
            T a6_value;
            if (k == 1) {
                a6_value = a6_second;
            } else if (k == nlev - 2) {
                a6_value = a6_penultimate;
            } else {
                const T qa = q[WOMBAT_VERTICAL_INDEX(k, lat)];
                a6_value = static_cast<T>(3) * (
                    qa + qa - (ar[k] + al[k])
                );
            }
            value = ar[k] + static_cast<T>(0.5) * courant * (
                al[k] - ar[k] +
                a6_value * (static_cast<T>(1) - r23 * courant)
            );
        } else {
            const int next_k = k + 1;
            T a6_value;
            if (next_k == nlev - 1) {
                a6_value = a6_bottom;
            } else if (next_k == nlev - 2) {
                a6_value = a6_penultimate;
            } else {
                const T qa = q[WOMBAT_VERTICAL_INDEX(next_k, lat)];
                a6_value = static_cast<T>(3) * (
                    qa + qa - (ar[next_k] + al[next_k])
                );
            }
            value = al[k + 1] + static_cast<T>(0.5) * courant * (
                al[k + 1] - ar[k + 1] -
                a6_value * (
                    static_cast<T>(1) + r23 * courant
                )
            );
        }
        const T flux = wz[k * ncol + col] * value;
        dq[WOMBAT_VERTICAL_INDEX(k, lat)] += previous_flux - flux;
        previous_flux = flux;
    }
    dq[WOMBAT_VERTICAL_INDEX(nlev - 1, lat)] += previous_flux;

    if (fill && lat >= 2 && lat <= nlat - 3) {
        int index = WOMBAT_VERTICAL_INDEX(0, lat);
        if (dq[index] < static_cast<T>(0)) {
            const int next = WOMBAT_VERTICAL_LEVEL_OFFSET(index, 0, 1);
            dq[next] += dq[index];
            dq[index] = static_cast<T>(0);
        }
        for (int k = 1; k < nlev - 1; ++k) {
            index = WOMBAT_VERTICAL_INDEX(k, lat);
            if (dq[index] < static_cast<T>(0)) {
                const int previous = WOMBAT_VERTICAL_LEVEL_OFFSET(
                    index, k, -1
                );
                const int next = WOMBAT_VERTICAL_LEVEL_OFFSET(index, k, 1);
                const T qup = dq[previous];
                const T qly = -dq[index];
                const T dup = wmin(qly, qup);
                dq[previous] = qup - dup;
                dq[index] = dup - qly;
                dq[next] += dq[index];
                dq[index] = static_cast<T>(0);
            }
        }
        index = WOMBAT_VERTICAL_INDEX(nlev - 1, lat);
        if (dq[index] < static_cast<T>(0)) {
            const int previous = WOMBAT_VERTICAL_LEVEL_OFFSET(
                index, nlev - 1, -1
            );
            const T qup = dq[previous];
            const T qly = -dq[index];
            const T dup = wmin(qly, qup);
            dq[previous] = qup - dup;
            dq[index] = static_cast<T>(0);
        }
    }

    if (finalize_output) {
        for (int k = 0; k < nlev; ++k) {
            const int index = WOMBAT_VERTICAL_INDEX(k, lat);
            T value = dq[index] / delp2[k * ncol + col];
            if (value < static_cast<T>(0)) {
                value = static_cast<T>(1.0e-26);
            }
            dq[index] = value;
            if (lat == 0) {
                dq[WOMBAT_VERTICAL_ROW_OFFSET(index, k, 1)] = value;
            } else if (lat == nlat - 1) {
                dq[WOMBAT_VERTICAL_ROW_OFFSET(index, k, nlat - 2)] = value;
            }
        }
    }
#undef WOMBAT_VERTICAL_ROW_OFFSET
#undef WOMBAT_VERTICAL_LEVEL_OFFSET
#undef WOMBAT_VERTICAL_INDEX
}
