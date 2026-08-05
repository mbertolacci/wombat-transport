template <typename T>
__global__ void apply_vdiff(
    const T* tracer_in,
    T* tracer_out,
    const T* cch,
    const T* zeh,
    const T* termh,
    const T* dry_mass,
    const T* area_m2,
    const T* cgs,
    const T* kvh,
    const T* potbar,
    const T* rpdel,
    const T* rrho,
    const T* tmp1,
    T dt_s,
    const int* start_level_value,
    const T* surface_flux,
    int has_flux,
    T* qmx,
    int* negative_count,
    int ntracer,
    int nlev,
    int nlat,
    int nlon,
    int nlane
) {
    const int work = blockDim.x * blockIdx.x + threadIdx.x;
    const int work_size = nlat * nlon * ntracer;
    if (work >= work_size) {
        return;
    }

    const int active_tracer = work % ntracer;
    const int lane = active_tracer % nlane;
    const int block = active_tracer / nlane;
    const int column = work / ntracer;
    const int lon = column % nlon;
    const int lat = column / nlon;
    const int horizontal = lat * nlon + lon;
    const int tracer_block_offset = block * nlev * nlat * nlon * nlane;
    const int flux_index =
        (block * nlat * nlon + horizontal) * nlane + lane;
    const int horizontal_size = nlat * nlon;
    const int start_level = start_level_value[0];
    const T ztodtgor = dt_s * static_cast<T>(9.80665) /
        static_cast<T>(287.0);
    const T surface_flux_value = has_flux
        ? surface_flux[flux_index]
        : static_cast<T>(0);
    const bool column_has_flux =
        surface_flux_value > static_cast<T>(0)
        || surface_flux_value < static_cast<T>(0);

    bool adjust = false;
    if (column_has_flux) {
        for (int lev = 0; lev < nlev; ++lev) {
            const int tracer_index =
                tracer_block_offset +
                ((lev * nlat + lat) * nlon + lon) * nlane + lane;
            qmx[tracer_index] = tracer_in[tracer_index];
        }
        for (int lev = start_level; lev < nlev; ++lev) {
            const int center = lev * horizontal_size + horizontal;
            const int edge_now = lev * horizontal_size + horizontal;
            const int edge_next = (lev + 1) * horizontal_size + horizontal;
            const int tracer_index = center * nlane + lane;
            const int block_tracer_index = tracer_block_offset + tracer_index;
            const T scale = ztodtgor * rpdel[center];
            const T term_next = potbar[edge_next] * kvh[edge_next];
            const T term_now = potbar[edge_now] * kvh[edge_now];
            const T flux_rrho = surface_flux_value * rrho[horizontal];
            const T cgq_next = flux_rrho * cgs[edge_next];
            const T cgq_now = flux_rrho * cgs[edge_now];
            const T value = tracer_in[block_tracer_index] +
                scale * (term_next * cgq_next - term_now * cgq_now);
            qmx[block_tracer_index] = value;
            if (value < static_cast<T>(0)) {
                adjust = true;
            }
        }
        if (adjust) {
            for (int lev = start_level; lev < nlev; ++lev) {
                const int tracer_index =
                    tracer_block_offset +
                    ((lev * nlat + lat) * nlon + lon) * nlane + lane;
                qmx[tracer_index] = tracer_in[tracer_index];
            }
        }
    }

    int tracer_index =
        tracer_block_offset + ((lat * nlon + lon) * nlane) + lane;
    int center = horizontal;
    T before_mass = tracer_in[tracer_index] * dry_mass[center];
    T source = column_has_flux ? qmx[tracer_index] : tracer_in[tracer_index];
    tracer_out[tracer_index] = source * termh[center];

    for (int lev = 1; lev < nlev - 1; ++lev) {
        tracer_index =
            tracer_block_offset +
            ((lev * nlat + lat) * nlon + lon) * nlane + lane;
        center = lev * horizontal_size + horizontal;
        const int previous =
            tracer_block_offset +
            (((lev - 1) * nlat + lat) * nlon + lon) * nlane + lane;
        before_mass += tracer_in[tracer_index] * dry_mass[center];
        source = column_has_flux ? qmx[tracer_index] : tracer_in[tracer_index];
        tracer_out[tracer_index] =
            (source + cch[center] * tracer_out[previous]) * termh[center];
    }

    const int last_lev = nlev - 1;
    tracer_index =
        tracer_block_offset +
        ((last_lev * nlat + lat) * nlon + lon) * nlane + lane;
    center = last_lev * horizontal_size + horizontal;
    const int previous =
        tracer_block_offset +
        (((last_lev - 1) * nlat + lat) * nlon + lon) * nlane + lane;
    const int penultimate_center =
        (last_lev - 1) * horizontal_size + horizontal;
    const T tmp1d = static_cast<T>(1) /
        (static_cast<T>(1) + cch[center] *
            (static_cast<T>(1) - zeh[penultimate_center]));
    before_mass += tracer_in[tracer_index] * dry_mass[center];
    source = column_has_flux ? qmx[tracer_index] : tracer_in[tracer_index];
    tracer_out[tracer_index] =
        (source +
         (column_has_flux ? surface_flux_value * tmp1[horizontal]
                   : static_cast<T>(0)) +
         cch[center] * tracer_out[previous]) *
        tmp1d;

    for (int lev = nlev - 2; lev >= 0; --lev) {
        tracer_index =
            tracer_block_offset +
            ((lev * nlat + lat) * nlon + lon) * nlane + lane;
        const int next =
            tracer_block_offset +
            (((lev + 1) * nlat + lat) * nlon + lon) * nlane + lane;
        center = lev * horizontal_size + horizontal;
        tracer_out[tracer_index] += zeh[center] * tracer_out[next];
    }

    T after_mass = static_cast<T>(0);
    int local_negative_count = 0;
    for (int lev = 0; lev < nlev; ++lev) {
        tracer_index =
            tracer_block_offset +
            ((lev * nlat + lat) * nlon + lon) * nlane + lane;
        center = lev * horizontal_size + horizontal;
        T value = tracer_out[tracer_index];
        if (value < static_cast<T>(0)) {
            ++local_negative_count;
            value = static_cast<T>(0);
            tracer_out[tracer_index] = value;
        }
        after_mass += value * dry_mass[center];
    }
    if (local_negative_count) {
        atomicAdd(negative_count, local_negative_count);
    }

    if (column_has_flux) {
        before_mass += surface_flux_value * area_m2[horizontal] * dt_s;
    }
    T ratio = static_cast<T>(1);
    const bool before_nonzero =
        before_mass > static_cast<T>(0) || before_mass < static_cast<T>(0);
    const bool after_nonzero =
        after_mass > static_cast<T>(0) || after_mass < static_cast<T>(0);
    if (before_nonzero && after_nonzero) {
        ratio = before_mass / after_mass;
    }
    for (int lev = 0; lev < nlev; ++lev) {
        tracer_index =
            tracer_block_offset +
            ((lev * nlat + lat) * nlon + lon) * nlane + lane;
        tracer_out[tracer_index] *= ratio;
    }
}
