template <typename T>
__global__ void apply_convection(
    T* tracer,
    T* diag,
    const T* cmfmc,
    const T* dtrain,
    const T* delp_hpa,
    const T* delp_dry,
    const T* bmass,
    const T* dqrcu,
    const T* reevapcn,
    const T* area,
    int diagnostics,
    int reconstruct_conv_precip_flux,
    int internal_steps,
    T internal_dt_s,
    int tracer_count,
    int nlev,
    int nlat,
    int nlon,
    int lane_width
) {
    const int work = blockDim.x * blockIdx.x + threadIdx.x;
    const int ncol = nlat * nlon;
    const int work_size = ncol * tracer_count;
    if (work >= work_size) {
        return;
    }

    const int active_tracer = work % tracer_count;
    const int col = work / tracer_count;
    const int block = active_tracer / lane_width;
    const int lane = active_tracer % lane_width;
    const int block_offset = block * nlev * ncol * lane_width;
    const int bottom = nlev - 1;
    const T tiny = static_cast<T>(1.0e-14);
    const T dns = static_cast<T>(internal_steps);

    int cloud_base = bottom;
    for (int level = bottom; level >= 0; --level) {
        const int center = level * ncol + col;
        T dqrcu_value = static_cast<T>(0);
        if (reconstruct_conv_precip_flux) {
            if (level == 0) {
                dqrcu_value = dqrcu[center] + reevapcn[center];
            } else if (level < bottom) {
                const int previous = (level - 1) * ncol + col;
                dqrcu_value = dqrcu[center] +
                    (reevapcn[center] * delp_hpa[center] -
                     reevapcn[previous] * delp_hpa[previous]) /
                    delp_hpa[center];
            }
        } else {
            dqrcu_value = dqrcu[center];
        }
        if (dqrcu_value > static_cast<T>(0)) {
            cloud_base = level;
            break;
        }
    }

    const bool mixes_below_base =
        cloud_base < bottom &&
        cmfmc[(cloud_base + 1) * ncol + col] > tiny;
    T mass_below_base = static_cast<T>(0);
    T cmfmc_base = static_cast<T>(0);
    T inv_denominator = static_cast<T>(1);
    T inv_denom_qc = static_cast<T>(1);
    if (mixes_below_base) {
        T denominator = static_cast<T>(0);
        for (int level = cloud_base + 1; level < nlev; ++level) {
            const int center = level * ncol + col;
            denominator += delp_dry[center];
            mass_below_base += bmass[center];
        }
        if (denominator <= static_cast<T>(0)) {
            denominator = static_cast<T>(1);
        }
        cmfmc_base = cmfmc[(cloud_base + 1) * ncol + col];
        const T denom_qc = mass_below_base + cmfmc_base * internal_dt_s;
        inv_denominator = static_cast<T>(1) / denominator;
        inv_denom_qc = static_cast<T>(1) / denom_qc;
    }

    for (int step = 0; step < internal_steps; ++step) {
        const int base_index =
            block_offset + (cloud_base * ncol + col) * lane_width + lane;
        T qc = tracer[base_index];

        if (mixes_below_base) {
            T qb_num = static_cast<T>(0);
            for (int level = cloud_base + 1; level < nlev; ++level) {
                const int center = level * ncol + col;
                const int tracer_index =
                    block_offset + center * lane_width + lane;
                qb_num += tracer[tracer_index] * delp_dry[center];
            }
            const T qb = qb_num * inv_denominator;
            qc = (
                mass_below_base * qb +
                cmfmc_base * tracer[base_index] * internal_dt_s
            ) * inv_denom_qc;
            for (int level = cloud_base + 1; level < nlev; ++level) {
                const int tracer_index =
                    block_offset + (level * ncol + col) * lane_width + lane;
                tracer[tracer_index] = qc;
            }
        }

        for (int level = cloud_base; level > 0; --level) {
            const int center = level * ncol + col;
            const int tracer_index =
                block_offset + center * lane_width + lane;
            const int above_index =
                block_offset + ((level - 1) * ncol + col) * lane_width + lane;
            const T cmfmc_below = level == bottom
                ? static_cast<T>(0)
                : cmfmc[(level + 1) * ncol + col];

            if (cmfmc_below > tiny) {
                const T cmfmc_current = cmfmc[center];
                const T cmout = cmfmc_current + dtrain[center];
                const T entrn = cmout - cmfmc_below;
                const bool entrains =
                    entrn >= static_cast<T>(0) &&
                    cmout > static_cast<T>(0);
                const T qc_pres = qc;
                const T current = tracer[tracer_index];
                T qc_next = qc_pres;
                if (entrains) {
                    qc_next = (
                        cmfmc_below * qc_pres + entrn * current
                    ) / cmout;
                }

                T delq = cmfmc_below * qc_pres;
                const T temp = -(cmfmc_current * qc_next);
                delq += temp;
                qc = qc_next;
                const T upward = cmfmc_current * tracer[above_index];
                delq += upward;
                if (diagnostics) {
                    diag[tracer_index] +=
                        (-temp - upward) * area[col] / dns;
                }
                delq -= cmfmc_below * current;
                delq *= internal_dt_s / bmass[center];
                if (current + delq < static_cast<T>(0)) {
                    delq = -current;
                }
                tracer[tracer_index] = current + delq;
            } else {
                qc = tracer[tracer_index];
                const T cmfmc_current = cmfmc[center];
                if (cmfmc_current > tiny) {
                    const T current = tracer[tracer_index];
                    T delq = -cmfmc_current * qc;
                    delq += cmfmc_current * tracer[above_index];
                    delq *= internal_dt_s / bmass[center];
                    if (current + delq < static_cast<T>(0)) {
                        delq = -current;
                    }
                    tracer[tracer_index] = current + delq;
                }
            }
        }
    }
}
