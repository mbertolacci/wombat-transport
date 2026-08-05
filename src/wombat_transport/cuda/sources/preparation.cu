__device__ void average_poles(
    double* pressure,
    const double* rel_area,
    int nlat,
    int nlon
) {
    double south = 0.0;
    double south_weight = 0.0;
    double north = 0.0;
    double north_weight = 0.0;
    for (int lat = 0; lat < 2; ++lat) {
        for (int lon = 0; lon < nlon; ++lon) {
            const int horizontal = lat * nlon + lon;
            south += pressure[horizontal] * rel_area[horizontal];
            south_weight += rel_area[horizontal];
        }
    }
    for (int lat = nlat - 2; lat < nlat; ++lat) {
        for (int lon = 0; lon < nlon; ++lon) {
            const int horizontal = lat * nlon + lon;
            north += pressure[horizontal] * rel_area[horizontal];
            north_weight += rel_area[horizontal];
        }
    }
    south /= south_weight;
    north /= north_weight;
    for (int lon = 0; lon < nlon; ++lon) {
        pressure[lon] = south;
        pressure[nlon + lon] = south;
        pressure[(nlat - 2) * nlon + lon] = north;
        pressure[(nlat - 1) * nlon + lon] = north;
    }
}

__device__ void tpcore_divergence(
    const double* xmass,
    const double* ymass,
    const double* geofac,
    double geofac_pc,
    double* output,
    bool bottom_reversed,
    int nlev,
    int nlat,
    int nlon
) {
    const int horizontal_size = nlat * nlon;
    const int j1p = 2;
    const int j2p = nlat - 3;
    for (int lev = 0; lev < nlev; ++lev) {
        const int target = bottom_reversed ? nlev - 1 - lev : lev;
        for (int horizontal = 0; horizontal < horizontal_size; ++horizontal) {
            output[target * horizontal_size + horizontal] = 0.0;
        }
        for (int lat = j1p; lat <= j2p; ++lat) {
            for (int lon = 0; lon < nlon; ++lon) {
                const int next_lon = lon == nlon - 1 ? 0 : lon + 1;
                const int center = (lev * nlat + lat) * nlon + lon;
                const int north = center + nlon;
                const int east = (lev * nlat + lat) * nlon + next_lon;
                output[target * horizontal_size + lat * nlon + lon] =
                    (ymass[center] - ymass[north]) * geofac[lat]
                    + xmass[center] - xmass[east];
            }
        }
        double south = 0.0;
        double north = 0.0;
        for (int lon = 0; lon < nlon; ++lon) {
            south += ymass[(lev * nlat + j1p) * nlon + lon];
            north += ymass[(lev * nlat + j2p + 1) * nlon + lon];
        }
        south = -(south / static_cast<double>(nlon)) * geofac_pc;
        north = (north / static_cast<double>(nlon)) * geofac_pc;
        for (int lon = 0; lon < nlon; ++lon) {
            output[(target * nlat) * nlon + lon] = south;
            output[(target * nlat + 1) * nlon + lon] = south;
            output[(target * nlat + nlat - 2) * nlon + lon] = north;
            output[(target * nlat + nlat - 1) * nlon + lon] = north;
        }
    }
}

extern "C" __global__ void prepare_surface_endpoints(
    const double* ps_start_pa,
    const double* ps_end_pa,
    const double* qv_start,
    const double* qv_end,
    const double* hyai,
    const double* hybi,
    double* wet_start,
    double* wet_end,
    double* dry_start,
    double* dry_end,
    int nlev,
    int nlat,
    int nlon
) {
    const int horizontal = blockDim.x * blockIdx.x + threadIdx.x;
    const int horizontal_size = nlat * nlon;
    if (horizontal >= horizontal_size) {
        return;
    }
    const double ps0 = ps_start_pa[horizontal] / 100.0;
    const double ps1 = ps_end_pa[horizontal] / 100.0;
    wet_start[horizontal] = ps0;
    wet_end[horizontal] = ps1;
    double dry0 = hyai[nlev];
    double dry1 = hyai[nlev];
    for (int lev = 0; lev < nlev; ++lev) {
        const double da = hyai[lev] - hyai[lev + 1];
        const double db = hybi[lev] - hybi[lev + 1];
        const int center = lev * horizontal_size + horizontal;
        dry0 += (da + db * ps0) * (1.0 - qv_start[center]);
        dry1 += (da + db * ps1) * (1.0 - qv_end[center]);
    }
    dry_start[horizontal] = dry0 < 0.0 ? ps0 : dry0;
    dry_end[horizontal] = dry1 < 0.0 ? ps1 : dry1;
}

extern "C" __global__ void average_surface_endpoint_poles(
    double* wet_start,
    double* wet_end,
    double* dry_start,
    double* dry_end,
    const double* area,
    int nlat,
    int nlon
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    average_poles(wet_start, area, nlat, nlon);
    average_poles(wet_end, area, nlat, nlon);
    average_poles(dry_start, area, nlat, nlon);
    average_poles(dry_end, area, nlat, nlon);
}

extern "C" __global__ void interpolate_step_meteorology(
    const double* wet_start_endpoint,
    const double* wet_end_endpoint,
    const double* dry_start_endpoint,
    const double* dry_end_endpoint,
    const double* qv_start,
    const double* qv_end,
    const double* temperature_start,
    const double* temperature_end,
    double start_fraction,
    double end_fraction,
    double midpoint_fraction,
    double* wet_start,
    double* wet_end,
    double* dry_start,
    double* dry_end,
    double* qv_midpoint,
    double* temperature_midpoint,
    int nlev,
    int nlat,
    int nlon
) {
    const int index = blockDim.x * blockIdx.x + threadIdx.x;
    const int horizontal_size = nlat * nlon;
    const int center_size = nlev * horizontal_size;
    if (index < horizontal_size) {
        const double wet_delta =
            wet_end_endpoint[index] - wet_start_endpoint[index];
        const double dry_delta =
            dry_end_endpoint[index] - dry_start_endpoint[index];
        wet_start[index] = wet_start_endpoint[index] + wet_delta * start_fraction;
        wet_end[index] = wet_start_endpoint[index] + wet_delta * end_fraction;
        dry_start[index] = dry_start_endpoint[index] + dry_delta * start_fraction;
        dry_end[index] = dry_start_endpoint[index] + dry_delta * end_fraction;
    }
    if (index < center_size) {
        qv_midpoint[index] =
            qv_start[index] + (qv_end[index] - qv_start[index]) * midpoint_fraction;
        temperature_midpoint[index] =
            temperature_start[index]
            + (temperature_end[index] - temperature_start[index]) * midpoint_fraction;
    }
}

extern "C" __global__ void prepare_tpcore_double(
    const double* p1_input,
    const double* p2_input,
    const double* u,
    const double* v,
    double dt_s,
    const double* rel_area,
    const double* geofac,
    double geofac_pc,
    const double* cose,
    const double* cosp,
    const double* dap_geos,
    const double* dbk_geos,
    const double* dap_top,
    const double* dbk_top,
    double* delp1,
    double* delpm,
    double* delp2,
    double* pu,
    double* xmass,
    double* ymass,
    double* vertical_mass_flux,
    double* normalized_vertical_courant,
    double* cx,
    double* cy,
    double* ua,
    double* va,
    long long* jn,
    long long* js,
    double* p1,
    double* p2,
    double* work3,
    double* work2,
    double* xfix,
    double* mmfd,
    double* mmf,
    double* fxintegral,
    int nlev,
    int nlat,
    int nlon
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    const int horizontal_size = nlat * nlon;
    const int j1p = 2;
    const int j2p = nlat - 3;
    double dgpress = 0.0;
    for (int lat = 0; lat < nlat; ++lat) {
        for (int lon = 0; lon < nlon; ++lon) {
            const int horizontal = lat * nlon + lon;
            p1[horizontal] = p1_input[horizontal];
            p2[horizontal] = p2_input[horizontal];
            dgpress +=
                (p2[horizontal] - p1[horizontal]) * rel_area[horizontal];
        }
    }
    for (int horizontal = 0; horizontal < horizontal_size; ++horizontal) {
        p2[horizontal] -= dgpress;
    }
    average_poles(p1, rel_area, nlat, nlon);
    average_poles(p2, rel_area, nlat, nlon);

    const double dlambda = 2.0 * 3.14159265358979323846 / nlon;
    const double dphi = 3.14159265358979323846 / (nlat - 1);
    for (int lev = 0; lev < nlev; ++lev) {
        for (int lat = 0; lat < nlat; ++lat) {
            for (int lon = 0; lon < nlon; ++lon) {
                const int center = (lev * nlat + lat) * nlon + lon;
                const int horizontal = lat * nlon + lon;
                work3[center] = dap_geos[lev] + dbk_geos[lev] * 0.5
                    * (p1[horizontal] + p2[horizontal]);
            }
        }
        for (int lat = 0; lat < nlat; ++lat) {
            const double factx =
                0.5 * dt_s / (dlambda * 6371007.2 * cosp[lat]);
            for (int lon = 0; lon < nlon; ++lon) {
                const int previous_lon = lon == 0 ? nlon - 1 : lon - 1;
                const int source = (lev * nlat + lat) * nlon + lon;
                const int previous =
                    (lev * nlat + lat) * nlon + previous_lon;
                const int target =
                    ((nlev - 1 - lev) * nlat + lat) * nlon + lon;
                xmass[target] = factx * (
                    u[source] * work3[source]
                    + u[previous] * work3[previous]
                );
            }
        }
        const double facty = 0.5 * dt_s / (6371007.2 * dphi);
        for (int lon = 0; lon < nlon; ++lon) {
            const int source = lev * horizontal_size + lon;
            const int target = (nlev - 1 - lev) * horizontal_size + lon;
            ymass[target] = facty * cose[0] * v[source] * work3[source];
        }
        for (int lat = 1; lat < nlat; ++lat) {
            for (int lon = 0; lon < nlon; ++lon) {
                const int source = (lev * nlat + lat) * nlon + lon;
                const int south = source - nlon;
                const int target =
                    ((nlev - 1 - lev) * nlat + lat) * nlon + lon;
                ymass[target] = facty * cose[lat] * (
                    v[source] * work3[source]
                    + v[south] * work3[south]
                );
            }
        }
    }

    tpcore_divergence(
        xmass,
        ymass,
        geofac,
        geofac_pc,
        work3,
        true,
        nlev,
        nlat,
        nlon
    );
    for (int lat = 0; lat < nlat; ++lat) {
        for (int lon = 0; lon < nlon; ++lon) {
            const int horizontal = lat * nlon + lon;
            double value = 0.0;
            for (int lev = 0; lev < nlev; ++lev) {
                value += work3[
                    (nlev - 1 - lev) * horizontal_size + horizontal
                ];
            }
            work2[horizontal] = value;
        }
    }
    dgpress = 0.0;
    for (int lat = 0; lat < nlat; ++lat) {
        for (int lon = 0; lon < nlon; ++lon) {
            const int horizontal = lat * nlon + lon;
            dgpress += (
                p2[horizontal] - p1[horizontal] - work2[horizontal]
            ) * rel_area[horizontal];
        }
    }
    for (int lat = 0; lat < nlat; ++lat) {
        double total = 0.0;
        for (int lon = 0; lon < nlon; ++lon) {
            const int horizontal = lat * nlon + lon;
            total += p2[horizontal] - p1[horizontal] - work2[horizontal];
        }
        const double mean = total / nlon;
        mmfd[lat] = -(mean - dgpress);
    }
    mmfd[0] = -(p2[0] - p1[0] - work2[0] - dgpress);
    mmfd[1] = -(
        p2[nlon] - p1[nlon] - work2[nlon] - dgpress
    );
    mmfd[nlat - 2] = -(
        p2[(nlat - 2) * nlon] - p1[(nlat - 2) * nlon]
        - work2[(nlat - 2) * nlon] - dgpress
    );
    mmfd[nlat - 1] = -(
        p2[(nlat - 1) * nlon] - p1[(nlat - 1) * nlon]
        - work2[(nlat - 1) * nlon] - dgpress
    );
    for (int lat = 0; lat < nlat; ++lat) {
        mmf[lat] = 0.0;
        for (int lon = 0; lon < nlon; ++lon) {
            xfix[lat * nlon + lon] = 0.0;
        }
    }
    mmf[j1p] = mmfd[0] / geofac_pc;
    for (int lat = j1p; lat <= j2p; ++lat) {
        mmf[lat + 1] = mmf[lat] + mmfd[lat] / geofac[lat];
        fxintegral[0] = 0.0;
        double total = 0.0;
        for (int lon = 0; lon < nlon; ++lon) {
            const int horizontal = lat * nlon + lon;
            const double ddps =
                p2[horizontal] - p1[horizontal] - work2[horizontal];
            fxintegral[lon + 1] =
                fxintegral[lon] - (ddps - dgpress) - mmfd[lat];
            total += fxintegral[lon + 1];
        }
        const double mean = total / nlon;
        for (int lon = 0; lon < nlon; ++lon) {
            xfix[lat * nlon + lon] = fxintegral[lon] - mean;
        }
    }
    for (int lev_top = 0; lev_top < nlev; ++lev_top) {
        const int lev_bottom = nlev - 1 - lev_top;
        for (int lat = 0; lat < nlat; ++lat) {
            for (int lon = 0; lon < nlon; ++lon) {
                const int center =
                    (lev_top * nlat + lat) * nlon + lon;
                xmass[center] += dbk_geos[lev_bottom]
                    * xfix[lat * nlon + lon];
            }
        }
        for (int lat = j1p; lat <= j2p + 1; ++lat) {
            for (int lon = 0; lon < nlon; ++lon) {
                const int center =
                    (lev_top * nlat + lat) * nlon + lon;
                ymass[center] += dbk_geos[lev_bottom] * mmf[lat];
            }
        }
    }

    for (int horizontal = 0; horizontal < horizontal_size; ++horizontal) {
        p1[horizontal] = p1_input[horizontal];
        p2[horizontal] = p2_input[horizontal];
    }
    average_poles(p1, rel_area, nlat, nlon);
    average_poles(p2, rel_area, nlat, nlon);
    for (int lev = 0; lev < nlev; ++lev) {
        for (int horizontal = 0; horizontal < horizontal_size; ++horizontal) {
            delp1[lev * horizontal_size + horizontal] =
                dap_top[lev] + dbk_top[lev] * p1[horizontal];
            delpm[lev * horizontal_size + horizontal] =
                dap_top[lev] + dbk_top[lev] * 0.5
                * (p1[horizontal] + p2[horizontal]);
            delp2[lev * horizontal_size + horizontal] =
                dap_top[lev] + dbk_top[lev] * p2[horizontal];
            pu[lev * horizontal_size + horizontal] = 0.0;
            cx[lev * horizontal_size + horizontal] = 0.0;
            cy[lev * horizontal_size + horizontal] = 0.0;
        }
        for (int lat = j1p; lat <= j2p; ++lat) {
            pu[(lev * nlat + lat) * nlon] = 0.5 * (
                delpm[(lev * nlat + lat) * nlon]
                + delpm[(lev * nlat + lat) * nlon + nlon - 1]
            );
            for (int lon = 1; lon < nlon; ++lon) {
                const int center = (lev * nlat + lat) * nlon + lon;
                pu[center] = 0.5 * (delpm[center] + delpm[center - 1]);
            }
            for (int lon = 0; lon < nlon; ++lon) {
                const int center = (lev * nlat + lat) * nlon + lon;
                cx[center] = xmass[center] / pu[center];
                cy[center] = ymass[center] / (
                    0.5 * cose[lat]
                    * (delpm[center] + delpm[center - nlon])
                );
            }
        }
        for (int lon = 0; lon < nlon; ++lon) {
            const int center = (lev * nlat + j2p + 1) * nlon + lon;
            cy[center] = ymass[center] / (
                0.5 * cose[j2p + 1]
                * (delpm[center] + delpm[center - nlon])
            );
        }
    }

    tpcore_divergence(
        xmass,
        ymass,
        geofac,
        geofac_pc,
        work3,
        false,
        nlev,
        nlat,
        nlon
    );
    for (int horizontal = 0; horizontal < horizontal_size; ++horizontal) {
        double total = 0.0;
        for (int lev = 0; lev < nlev; ++lev) {
            total += work3[lev * horizontal_size + horizontal];
        }
        work2[horizontal] = total;
        vertical_mass_flux[horizontal] =
            work3[horizontal] - dbk_top[0] * work2[horizontal];
        normalized_vertical_courant[horizontal] =
            vertical_mass_flux[horizontal] > 0.0
            ? vertical_mass_flux[horizontal] / delp1[horizontal]
            : vertical_mass_flux[horizontal]
                / delp1[horizontal_size + horizontal];
        for (int lev = 1; lev < nlev - 1; ++lev) {
            const int center = lev * horizontal_size + horizontal;
            const int previous = center - horizontal_size;
            vertical_mass_flux[center] =
                vertical_mass_flux[previous] + work3[center]
                - dbk_top[lev] * work2[horizontal];
            normalized_vertical_courant[center] =
                vertical_mass_flux[center] > 0.0
                ? vertical_mass_flux[center] / delp1[center]
                : vertical_mass_flux[center]
                    / delp1[center + horizontal_size];
        }
        const int top = (nlev - 1) * horizontal_size + horizontal;
        vertical_mass_flux[top] = 0.0;
        normalized_vertical_courant[top] = 0.0;
    }

    for (int center = 0; center < nlev * horizontal_size; ++center) {
        ua[center] = 0.0;
        va[center] = 0.0;
    }
    for (int lev = 0; lev < nlev; ++lev) {
        for (int lat = j1p; lat <= j2p; ++lat) {
            for (int lon = 0; lon < nlon - 1; ++lon) {
                const int center = (lev * nlat + lat) * nlon + lon;
                ua[center] = 0.5 * (cx[center] + cx[center + 1]);
            }
            const int last = (lev * nlat + lat) * nlon + nlon - 1;
            ua[last] = 0.5 * (cx[last] + cx[last - nlon + 1]);
        }
        for (int lat = 1; lat < nlat - 1; ++lat) {
            for (int lon = 0; lon < nlon; ++lon) {
                const int center = (lev * nlat + lat) * nlon + lon;
                va[center] = 0.5 * (cy[center] + cy[center + nlon]);
            }
        }
    }
    const int js0 = (nlat + 1) / 2 - 1;
    const int jn0 = nlat - (js0 + 1);
    for (int lev = 0; lev < nlev; ++lev) {
        long long js_value = j1p;
        for (int lat = js0 < nlat - 1 ? js0 : nlat - 2;
             lat >= j1p;
             --lat) {
            bool found = false;
            for (int lon = 0; lon < nlon; ++lon) {
                const double value = cx[(lev * nlat + lat) * nlon + lon];
                if (value > 1.0 || value < -1.0) {
                    found = true;
                    break;
                }
            }
            if (found) {
                js_value = lat;
                break;
            }
        }
        long long jn_value = j2p;
        for (int lat = jn0 > 0 ? jn0 : 0;
             lat <= (j2p < nlat - 1 ? j2p : nlat - 2);
             ++lat) {
            bool found = false;
            for (int lon = 0; lon < nlon; ++lon) {
                const double value = cx[(lev * nlat + lat) * nlon + lon];
                if (value > 1.0 || value < -1.0) {
                    found = true;
                    break;
                }
            }
            if (found) {
                jn_value = lat;
                break;
            }
        }
        js[lev] = js_value;
        jn[lev] = jn_value;
    }
}

extern "C" __global__ void tpcore_prepare_pressure_delta_terms(
    const double* p1,
    const double* p2,
    const double* work2,
    const double* rel_area,
    double* terms,
    int subtract_work2,
    int horizontal_size
) {
    const int horizontal = blockDim.x * blockIdx.x + threadIdx.x;
    if (horizontal >= horizontal_size) {
        return;
    }
    double value = p2[horizontal] - p1[horizontal];
    if (subtract_work2) {
        value -= work2[horizontal];
    }
    terms[horizontal] = value * rel_area[horizontal];
}

extern "C" __global__ void tpcore_sum_pressure_delta(
    const double* terms,
    double* pressure_delta,
    int horizontal_size
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    double total = 0.0;
    for (int horizontal = 0; horizontal < horizontal_size; ++horizontal) {
        total += terms[horizontal];
    }
    pressure_delta[0] = total;
}

extern "C" __global__ void tpcore_apply_pressure_fix(
    const double* p1_input,
    const double* p2_input,
    const double* pressure_delta,
    double* p1,
    double* p2,
    int nlat,
    int nlon
) {
    const int horizontal = blockDim.x * blockIdx.x + threadIdx.x;
    const int horizontal_size = nlat * nlon;
    if (horizontal >= horizontal_size) {
        return;
    }
    p1[horizontal] = p1_input[horizontal];
    p2[horizontal] = p2_input[horizontal] - pressure_delta[0];
}

extern "C" __global__ void tpcore_average_pressure_poles(
    double* p1,
    double* p2,
    const double* rel_area,
    int nlat,
    int nlon
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    average_poles(p1, rel_area, nlat, nlon);
    average_poles(p2, rel_area, nlat, nlon);
}

extern "C" __global__ void tpcore_prepare_mass_flux(
    const double* p1,
    const double* p2,
    const double* u,
    const double* v,
    double dt_s,
    const double* cosp,
    const double* cose,
    const double* dap_geos,
    const double* dbk_geos,
    double* work3,
    double* xmass,
    double* ymass,
    int nlev,
    int nlat,
    int nlon
) {
    const int center = blockDim.x * blockIdx.x + threadIdx.x;
    const int horizontal_size = nlat * nlon;
    if (center >= nlev * horizontal_size) {
        return;
    }
    const int lev = center / horizontal_size;
    const int horizontal = center - lev * horizontal_size;
    const int lat = horizontal / nlon;
    const int lon = horizontal - lat * nlon;
    const int previous_lon = lon == 0 ? nlon - 1 : lon - 1;
    const int previous_horizontal = lat * nlon + previous_lon;
    const double midpoint_pressure = dap_geos[lev] + dbk_geos[lev] * 0.5
        * (p1[horizontal] + p2[horizontal]);
    const double previous_midpoint = dap_geos[lev] + dbk_geos[lev] * 0.5
        * (p1[previous_horizontal] + p2[previous_horizontal]);
    work3[center] = midpoint_pressure;
    const double dlambda = 2.0 * 3.14159265358979323846 / nlon;
    const double factx = 0.5 * dt_s / (dlambda * 6371007.2 * cosp[lat]);
    const int target =
        ((nlev - 1 - lev) * nlat + lat) * nlon + lon;
    xmass[target] = factx * (
        u[center] * midpoint_pressure
        + u[lev * horizontal_size + previous_horizontal]
            * previous_midpoint
    );
    const double dphi = 3.14159265358979323846 / (nlat - 1);
    const double facty = 0.5 * dt_s / (6371007.2 * dphi);
    if (lat == 0) {
        ymass[target] =
            facty * cose[0] * v[center] * midpoint_pressure;
    } else {
        const int south = center - nlon;
        const int south_horizontal = horizontal - nlon;
        const double south_midpoint =
            dap_geos[lev] + dbk_geos[lev] * 0.5
            * (p1[south_horizontal] + p2[south_horizontal]);
        ymass[target] = facty * cose[lat] * (
            v[center] * midpoint_pressure + v[south] * south_midpoint
        );
    }
}

extern "C" __global__ void tpcore_divergence_interior(
    const double* xmass,
    const double* ymass,
    const double* geofac,
    double* output,
    int bottom_reversed,
    int nlev,
    int nlat,
    int nlon
) {
    const int center = blockDim.x * blockIdx.x + threadIdx.x;
    const int horizontal_size = nlat * nlon;
    if (center >= nlev * horizontal_size) {
        return;
    }
    const int lev = center / horizontal_size;
    const int horizontal = center - lev * horizontal_size;
    const int lat = horizontal / nlon;
    if (lat < 2 || lat > nlat - 3) {
        return;
    }
    const int lon = horizontal - lat * nlon;
    const int next_lon = lon == nlon - 1 ? 0 : lon + 1;
    const int target_lev = bottom_reversed ? nlev - 1 - lev : lev;
    output[target_lev * horizontal_size + horizontal] =
        (ymass[center] - ymass[center + nlon]) * geofac[lat]
        + xmass[center]
        - xmass[lev * horizontal_size + lat * nlon + next_lon];
}

extern "C" __global__ void tpcore_divergence_poles(
    const double* ymass,
    double geofac_pc,
    double* output,
    int bottom_reversed,
    int nlev,
    int nlat,
    int nlon
) {
    const int lev = blockDim.x * blockIdx.x + threadIdx.x;
    if (lev >= nlev) {
        return;
    }
    const int horizontal_size = nlat * nlon;
    const int target_lev = bottom_reversed ? nlev - 1 - lev : lev;
    double south = 0.0;
    double north = 0.0;
    for (int lon = 0; lon < nlon; ++lon) {
        south += ymass[(lev * nlat + 2) * nlon + lon];
        north += ymass[(lev * nlat + nlat - 2) * nlon + lon];
    }
    south = -(south / static_cast<double>(nlon)) * geofac_pc;
    north = (north / static_cast<double>(nlon)) * geofac_pc;
    for (int lon = 0; lon < nlon; ++lon) {
        output[(target_lev * nlat) * nlon + lon] = south;
        output[(target_lev * nlat + 1) * nlon + lon] = south;
        output[(target_lev * nlat + nlat - 2) * nlon + lon] = north;
        output[(target_lev * nlat + nlat - 1) * nlon + lon] = north;
    }
}

extern "C" __global__ void tpcore_sum_vertical(
    const double* values,
    double* totals,
    int reversed,
    int nlev,
    int horizontal_size
) {
    const int horizontal = blockDim.x * blockIdx.x + threadIdx.x;
    if (horizontal >= horizontal_size) {
        return;
    }
    double total = 0.0;
    for (int lev = 0; lev < nlev; ++lev) {
        const int source_lev = reversed ? nlev - 1 - lev : lev;
        total += values[source_lev * horizontal_size + horizontal];
    }
    totals[horizontal] = total;
}

extern "C" __global__ void tpcore_prepare_pressure_rows(
    const double* p1,
    const double* p2,
    const double* work2,
    const double* pressure_delta,
    double* xfix,
    double* mmfd,
    double* mmf,
    int nlat,
    int nlon
) {
    const int lat = blockDim.x * blockIdx.x + threadIdx.x;
    if (lat >= nlat) {
        return;
    }
    const double dgpress = pressure_delta[0];
    double total = 0.0;
    for (int lon = 0; lon < nlon; ++lon) {
        const int horizontal = lat * nlon + lon;
        total += p2[horizontal] - p1[horizontal] - work2[horizontal];
        xfix[horizontal] = 0.0;
    }
    mmfd[lat] = -(total / nlon - dgpress);
    mmf[lat] = 0.0;
    if (lat < 2 || lat >= nlat - 2) {
        const int horizontal = lat * nlon;
        mmfd[lat] = -(
            p2[horizontal] - p1[horizontal] - work2[horizontal] - dgpress
        );
    }
}

extern "C" __global__ void tpcore_prepare_meridional_correction(
    const double* mmfd,
    const double* geofac,
    double geofac_pc,
    double* mmf,
    int nlat
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    mmf[2] = mmfd[0] / geofac_pc;
    for (int lat = 2; lat <= nlat - 3; ++lat) {
        mmf[lat + 1] = mmf[lat] + mmfd[lat] / geofac[lat];
    }
}

extern "C" __global__ void tpcore_prepare_zonal_correction(
    const double* p1,
    const double* p2,
    const double* work2,
    const double* pressure_delta,
    const double* mmfd,
    double* xfix,
    int nlat,
    int nlon
) {
    const int lat = blockDim.x * blockIdx.x + threadIdx.x + 2;
    if (lat > nlat - 3) {
        return;
    }
    const double dgpress = pressure_delta[0];
    double integral = 0.0;
    double total = 0.0;
    for (int lon = 0; lon < nlon; ++lon) {
        const int horizontal = lat * nlon + lon;
        const double ddps =
            p2[horizontal] - p1[horizontal] - work2[horizontal];
        integral = integral - (ddps - dgpress) - mmfd[lat];
        total += integral;
    }
    const double mean = total / nlon;
    integral = 0.0;
    for (int lon = 0; lon < nlon; ++lon) {
        const int horizontal = lat * nlon + lon;
        xfix[horizontal] = integral - mean;
        const double ddps =
            p2[horizontal] - p1[horizontal] - work2[horizontal];
        integral = integral - (ddps - dgpress) - mmfd[lat];
    }
}

extern "C" __global__ void tpcore_apply_pressure_correction(
    double* xmass,
    double* ymass,
    const double* xfix,
    const double* mmf,
    const double* dbk_geos,
    int nlev,
    int nlat,
    int nlon
) {
    const int center = blockDim.x * blockIdx.x + threadIdx.x;
    const int horizontal_size = nlat * nlon;
    if (center >= nlev * horizontal_size) {
        return;
    }
    const int lev = center / horizontal_size;
    const int horizontal = center - lev * horizontal_size;
    const int lat = horizontal / nlon;
    const double dbk = dbk_geos[nlev - 1 - lev];
    xmass[center] += dbk * xfix[horizontal];
    if (lat >= 2 && lat <= nlat - 2) {
        ymass[center] += dbk * mmf[lat];
    }
}

extern "C" __global__ void tpcore_copy_pressure(
    const double* p1_input,
    const double* p2_input,
    double* p1,
    double* p2,
    int nlat,
    int nlon
) {
    const int horizontal = blockDim.x * blockIdx.x + threadIdx.x;
    const int horizontal_size = nlat * nlon;
    if (horizontal >= horizontal_size) {
        return;
    }
    p1[horizontal] = p1_input[horizontal];
    p2[horizontal] = p2_input[horizontal];
}

extern "C" __global__ void tpcore_prepare_pressure_terms(
    const double* p1,
    const double* p2,
    const double* xmass,
    const double* ymass,
    const double* cose,
    const double* dap_top,
    const double* dbk_top,
    double* delp1,
    double* delpm,
    double* delp2,
    double* pu,
    double* cx,
    double* cy,
    long long* jn,
    long long* js,
    int nlev,
    int nlat,
    int nlon
) {
    const int center = blockDim.x * blockIdx.x + threadIdx.x;
    const int horizontal_size = nlat * nlon;
    if (center >= nlev * horizontal_size) {
        return;
    }
    const int lev = center / horizontal_size;
    const int horizontal = center - lev * horizontal_size;
    const int lat = horizontal / nlon;
    const int lon = horizontal - lat * nlon;
    const double pmid = dap_top[lev] + dbk_top[lev] * 0.5
        * (p1[horizontal] + p2[horizontal]);
    delp1[center] = dap_top[lev] + dbk_top[lev] * p1[horizontal];
    delpm[center] = pmid;
    delp2[center] = dap_top[lev] + dbk_top[lev] * p2[horizontal];
    pu[center] = 0.0;
    cx[center] = 0.0;
    cy[center] = 0.0;
    if (lat >= 2 && lat <= nlat - 3) {
        const int previous_lon = lon == 0 ? nlon - 1 : lon - 1;
        const int west_horizontal = lat * nlon + previous_lon;
        const double west_pmid = dap_top[lev] + dbk_top[lev] * 0.5
            * (p1[west_horizontal] + p2[west_horizontal]);
        const double pu_value = 0.5 * (pmid + west_pmid);
        pu[center] = pu_value;
        const double cx_value = xmass[center] / pu_value;
        cx[center] = cx_value;
        if (cx_value > 1.0 || cx_value < -1.0) {
            const int js0 = (nlat + 1) / 2 - 1;
            const int jn0 = nlat - (js0 + 1);
            if (lat <= js0) {
                atomicMax(
                    reinterpret_cast<unsigned long long*>(&js[lev]),
                    static_cast<unsigned long long>(lat)
                );
            }
            if (lat >= jn0) {
                atomicMin(
                    reinterpret_cast<unsigned long long*>(&jn[lev]),
                    static_cast<unsigned long long>(lat)
                );
            }
        }
        const int south_horizontal = horizontal - nlon;
        const double south_pmid = dap_top[lev] + dbk_top[lev] * 0.5
            * (p1[south_horizontal] + p2[south_horizontal]);
        cy[center] = ymass[center] / (
            0.5 * cose[lat] * (pmid + south_pmid)
        );
    } else if (lat == nlat - 2) {
        const int south_horizontal = horizontal - nlon;
        const double south_pmid = dap_top[lev] + dbk_top[lev] * 0.5
            * (p1[south_horizontal] + p2[south_horizontal]);
        cy[center] = ymass[center] / (
            0.5 * cose[lat] * (pmid + south_pmid)
        );
    }
}

extern "C" __global__ void tpcore_prepare_vertical_flux(
    const double* divergence,
    const double* dbk_top,
    const double* delp1,
    double* work2,
    double* vertical_mass_flux,
    double* normalized_vertical_courant,
    int nlev,
    int horizontal_size
) {
    const int horizontal = blockDim.x * blockIdx.x + threadIdx.x;
    if (horizontal >= horizontal_size) {
        return;
    }
    double total = 0.0;
    for (int lev = 0; lev < nlev; ++lev) {
        total += divergence[lev * horizontal_size + horizontal];
    }
    work2[horizontal] = total;
    double flux = divergence[horizontal] - dbk_top[0] * total;
    vertical_mass_flux[horizontal] = flux;
    normalized_vertical_courant[horizontal] =
        flux > 0.0 ? flux / delp1[horizontal]
                   : flux / delp1[horizontal_size + horizontal];
    for (int lev = 1; lev < nlev - 1; ++lev) {
        const int center = lev * horizontal_size + horizontal;
        flux = flux + divergence[center] - dbk_top[lev] * total;
        vertical_mass_flux[center] = flux;
        normalized_vertical_courant[center] =
            flux > 0.0 ? flux / delp1[center]
                       : flux / delp1[center + horizontal_size];
    }
    const int top = (nlev - 1) * horizontal_size + horizontal;
    vertical_mass_flux[top] = 0.0;
    normalized_vertical_courant[top] = 0.0;
}

extern "C" __global__ void tpcore_prepare_cross_terms(
    const double* cx,
    const double* cy,
    double* ua,
    double* va,
    int nlev,
    int nlat,
    int nlon
) {
    const int center = blockDim.x * blockIdx.x + threadIdx.x;
    const int horizontal_size = nlat * nlon;
    if (center >= nlev * horizontal_size) {
        return;
    }
    const int horizontal = center % horizontal_size;
    const int lat = horizontal / nlon;
    const int lon = horizontal - lat * nlon;
    ua[center] = 0.0;
    va[center] = 0.0;
    if (lat >= 2 && lat <= nlat - 3) {
        const int east = lon == nlon - 1 ? center - nlon + 1 : center + 1;
        ua[center] = 0.5 * (cx[center] + cx[east]);
    }
    if (lat >= 1 && lat < nlat - 1) {
        va[center] = 0.5 * (cy[center] + cy[center + nlon]);
    }
}

extern "C" __global__ void tpcore_initialize_jn_js(
    long long* jn,
    long long* js,
    int nlev,
    int nlat
) {
    const int lev = blockDim.x * blockIdx.x + threadIdx.x;
    if (lev >= nlev) {
        return;
    }
    js[lev] = 2;
    jn[lev] = nlat - 3;
}

template <typename T>
__global__ void cast_plan_array(
    const double* source,
    T* destination,
    long long count
) {
    const long long index =
        static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;
    if (index < count) {
        destination[index] = static_cast<T>(source[index]);
    }
}

extern "C" __global__ void compute_vdiff_start_level(
    const double* wet_surface_pressure,
    const double* hyai,
    const double* hybi,
    int* start_level,
    int nlev,
    int nlat,
    int nlon
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    const int horizontal_size = nlat * nlon;
    double surface_pressure_total = 0.0;
    for (int horizontal = 0; horizontal < horizontal_size; ++horizontal) {
        surface_pressure_total += wet_surface_pressure[horizontal];
    }
    const double mean_surface_pressure =
        surface_pressure_total / static_cast<double>(horizontal_size);
    int break_index = 0;
    for (int top = nlev - 1; top >= 0; --top) {
        const int bottom = nlev - 1 - top;
        const double edge_lower =
            hyai[bottom] + hybi[bottom] * mean_surface_pressure;
        const double edge_upper =
            hyai[bottom + 1] + hybi[bottom + 1] * mean_surface_pressure;
        const double mean = 0.5 * (edge_lower + edge_upper);
        if (mean < 400.0) {
            break_index = top;
            break;
        }
    }
    int npbl = nlev - (break_index + 1);
    if (npbl < 1) {
        npbl = 1;
    }
    start_level[0] = nlev - npbl;
}

extern "C" __global__ void prepare_vdiff_double(
    const double* u_bottom,
    const double* v_bottom,
    const double* temperature_bottom,
    const double* sphu_bottom,
    const double* wet_surface_pressure,
    const double* pblh,
    const double* hflux,
    const double* eflux,
    const double* ustar,
    const double* area,
    const double* hyai,
    const double* hybi,
    const double* delp_dry_top,
    double dt_s,
    const int* start_level,
    double* plan_cch,
    double* plan_zeh,
    double* plan_termh,
    double* plan_cgs,
    double* plan_kvh,
    double* plan_potbar,
    double* plan_rpdel,
    double* plan_rrho,
    double* plan_tmp1,
    double* dry_mass_top,
    double* sphu_after_top,
    int nlev,
    int nlat,
    int nlon
) {
    const int horizontal = blockDim.x * blockIdx.x + threadIdx.x;
    const int horizontal_size = nlat * nlon;
    if (horizontal >= horizontal_size) {
        return;
    }
    const int lat = horizontal / nlon;
    const int lon = horizontal - lat * nlon;
    const double ps = wet_surface_pressure[horizontal];
    double pmid[47];
    double pint[48];
    double rpdel[47];
    double rpdeli[47];
    double zm[47];
    double thp[47];
    double virtual_temperature[47];
    double bxheight[47];
    double kvf[48];
    double kvh[48];
    double cgsh[48];
    double cgs[48];
    double potbar[48];
    double cah[47];
    double cch[47];
    double zeh[47];
    double termh[47];
    double shmx[47];
    double zfq[47];
    double sphu_diffused[47];

    for (int edge_top = 0; edge_top <= nlev; ++edge_top) {
        const int edge_bottom = nlev - edge_top;
        pint[edge_top] =
            (hyai[edge_bottom] + hybi[edge_bottom] * ps) * 100.0;
        kvf[edge_top] = 0.0;
        kvh[edge_top] = 0.0;
        cgsh[edge_top] = 0.0;
        cgs[edge_top] = 0.0;
        potbar[edge_top] = 0.0;
    }
    for (int top = 0; top < nlev; ++top) {
        const int bottom = nlev - 1 - top;
        const int source = bottom * horizontal_size + horizontal;
        const double q = sphu_bottom[source];
        const double temperature = temperature_bottom[source];
        const double water_vapor =
            28.9644 * q / (18.016 * (1.0 - q));
        const double xh2o = water_vapor / (1.0 + water_vapor);
        virtual_temperature[top] = temperature / (
            1.0 - xh2o * (1.0 - 18.016 / 28.9644)
        );
        const double edge_lower =
            hyai[bottom] + hybi[bottom] * ps;
        const double edge_upper =
            hyai[bottom + 1] + hybi[bottom + 1] * ps;
        pmid[top] = 0.5 * (edge_lower + edge_upper) * 100.0;
        bxheight[top] = 287.0 / 9.80665 * virtual_temperature[top]
            * log(edge_lower / edge_upper);
        rpdel[top] = 1.0 / (pint[top + 1] - pint[top]);
        rpdeli[top] = 0.0;
        cch[top] = 0.0;
        zeh[top] = 0.0;
        termh[top] = 0.0;
        cah[top] = 0.0;
        dry_mass_top[top * horizontal_size + horizontal] =
            delp_dry_top[top * horizontal_size + horizontal]
            * 100.0 / 9.80665 * area[horizontal];
    }
    for (int top = 0; top < nlev - 1; ++top) {
        rpdeli[top] = 1.0 / (pmid[top + 1] - pmid[top]);
    }
    double cumulative_height = 0.0;
    for (int lev_down = nlev - 1; lev_down >= 0; --lev_down) {
        cumulative_height += bxheight[lev_down];
        const int source =
            (nlev - 1 - lev_down) * horizontal_size + horizontal;
        const double temperature = temperature_bottom[source];
        thp[lev_down] = temperature
            * pow(1.0e5 / pmid[lev_down], 287.0 / 1004.64);
        zm[lev_down] = cumulative_height
            - log(pmid[lev_down] / pint[lev_down])
            * (287.0 / 9.80665) * virtual_temperature[lev_down];
    }

    const double water_flux = eflux[horizontal] / 2.5104e6;
    const double tmp1 = dt_s * 9.80665 * rpdel[nlev - 1];
    const double dshbot = water_flux * tmp1;
    for (int lev = 0; lev < nlev - 1; ++lev) {
        const int bottom = nlev - 1 - lev;
        const int next_bottom = nlev - 2 - lev;
        const int center = bottom * horizontal_size + horizontal;
        const int next = next_bottom * horizontal_size + horizontal;
        double dvdz2 =
            (u_bottom[center] - u_bottom[next])
            * (u_bottom[center] - u_bottom[next]);
        dvdz2 +=
            (v_bottom[center] - v_bottom[next])
            * (v_bottom[center] - v_bottom[next]);
        if (dvdz2 < 1.0e-36) {
            dvdz2 = 1.0e-36;
        }
        const double dz = zm[lev] - zm[lev + 1];
        dvdz2 /= dz * dz;
        const double thv_lev = thp[lev]
            * (1.0 + (461.0 / 287.0 - 1.0) * sphu_bottom[center]);
        const double thv_next = thp[lev + 1]
            * (1.0 + (461.0 / 287.0 - 1.0) * sphu_bottom[next]);
        const double sstab = 9.80665 * 2.0 * (thv_lev - thv_next)
            / ((thv_lev + thv_next) * dz);
        const double rinub = sstab / dvdz2;
        double fstab = 1.0
            / (1.0 + 10.0 * rinub * (1.0 + 8.0 * rinub));
        if (rinub < 0.0) {
            double funst = 1.0 - 18.0 * rinub;
            if (funst < 0.0) {
                funst = 0.0;
            }
            fstab = sqrt(funst);
        }
        const double ml2 = lev > 0 ? 900.0 : 0.0;
        double value = ml2 * sqrt(dvdz2) * fstab;
        if (value < 0.01) {
            value = 0.01;
        }
        kvf[lev + 1] = value;
    }
    for (int edge = 0; edge <= nlev; ++edge) {
        kvh[edge] = kvf[edge];
    }

    const int surface_source = horizontal;
    const double surface_temperature = temperature_bottom[surface_source];
    const double surface_q = sphu_bottom[surface_source];
    const double rrho = 287.0 * surface_temperature / pmid[nlev - 1];
    const double khfs = hflux[horizontal] * rrho / 1004.64;
    const double kshfs = water_flux * rrho;
    const double thvsrf = thp[nlev - 1] * (1.0 + 0.61 * surface_q);
    const double heatv = khfs + 0.61 * thp[nlev - 1] * kshfs;
    const double sign_heat = heatv < 0.0 ? -1.0e-10 : 1.0e-10;
    const double ustar_value = ustar[horizontal];
    const double pblh_value = pblh[horizontal];
    const double obklen =
        -thvsrf * ustar_value * ustar_value * ustar_value
        / (9.80665 * 0.4 * (heatv + sign_heat));
    const double fak1 = ustar_value * pblh_value * 0.4;
    double phiminv = 0.0;
    double phihinv = 0.0;
    double wm = 0.0;
    double fak2 = 0.0;
    double fak3 = 0.0;
    if (heatv > 0.0) {
        phiminv = pow(1.0 - 1.5 * pblh_value / obklen, 1.0 / 3.0);
        phihinv = sqrt(1.0 - 1.5 * pblh_value / obklen);
        wm = ustar_value * phiminv;
        fak2 = wm * pblh_value * 0.4;
        const double wstr = pow(
            heatv * 9.80665 * pblh_value / thvsrf,
            1.0 / 3.0
        );
        fak3 = 7.20 * wstr / wm;
    }

    const int start = start_level[0];
    for (int lev = nlev - 1; lev > start; --lev) {
        const double zm_value = zm[lev];
        if (zm_value >= pblh_value) {
            continue;
        }
        const double zp = zm[lev - 1];
        const double zmzp = 0.5 * (zm_value + zp);
        const double zh = zmzp / pblh_value;
        const double zl = zmzp / obklen;
        double zzh = 0.0;
        if (zh <= 1.0) {
            zzh = (1.0 - zh) * (1.0 - zh);
        }
        double pblk = 0.0;
        double pr = 0.0;
        if (heatv <= 0.0) {
            pblk = zl <= 1.0
                ? fak1 * zh * zzh / (1.0 + 5.0 * zl)
                : fak1 * zh * zzh / (5.0 + zl);
            kvh[lev] = pblk > kvf[lev] ? pblk : kvf[lev];
        } else {
            if (zh < 0.1) {
                const double term = pow(1.0 - 15.0 * zl, 1.0 / 3.0);
                pblk = fak1 * zh * zzh * term;
                pr = term / sqrt(1.0 - 15.0 * zl);
            } else {
                pblk = fak2 * zh * zzh;
                cgs[lev] = fak3 / (pblh_value * wm);
                pr = phiminv / phihinv
                    + (8.50 * 0.1 * 0.4) * fak3 / 8.50;
                cgsh[lev] = kshfs * cgs[lev];
            }
            const double kh = pblk / pr;
            kvh[lev] = kh > kvf[lev] ? kh : kvf[lev];
        }
    }

    potbar[0] = 0.0;
    for (int lev = 1; lev < nlev; ++lev) {
        const int bottom = nlev - 1 - lev;
        const int previous_bottom = bottom + 1;
        const double temperature = temperature_bottom[
            bottom * horizontal_size + horizontal
        ];
        const double previous_temperature = temperature_bottom[
            previous_bottom * horizontal_size + horizontal
        ];
        potbar[lev] = pint[lev]
            / (0.5 * (temperature + previous_temperature));
    }
    potbar[nlev] = pint[nlev] / surface_temperature;

    bool restore = false;
    for (int lev = 0; lev < nlev; ++lev) {
        const int bottom = nlev - 1 - lev;
        shmx[lev] = sphu_bottom[bottom * horizontal_size + horizontal];
    }
    if (start < nlev - 1) {
        const double ztodtgor = dt_s * 9.80665 / 287.0;
        for (int lev = start; lev < nlev; ++lev) {
            const int bottom = nlev - 1 - lev;
            const double source_q =
                sphu_bottom[bottom * horizontal_size + horizontal];
            const double scale = ztodtgor * rpdel[lev];
            shmx[lev] = source_q + scale * (
                potbar[lev + 1] * kvh[lev + 1] * cgsh[lev + 1]
                - potbar[lev] * kvh[lev] * cgsh[lev]
            );
            if (shmx[lev] < 1.0e-12) {
                restore = true;
            }
        }
        if (restore) {
            for (int lev = start; lev < nlev; ++lev) {
                const int bottom = nlev - 1 - lev;
                shmx[lev] =
                    sphu_bottom[bottom * horizontal_size + horizontal];
            }
        }
    }

    const double gorsq = (9.80665 / 287.0) * (9.80665 / 287.0);
    for (int lev = 0; lev < nlev - 1; ++lev) {
        const double tmp2 = dt_s * gorsq * rpdeli[lev]
            * potbar[lev + 1] * potbar[lev + 1];
        cah[lev] = kvh[lev + 1] * tmp2 * rpdel[lev];
        cch[lev + 1] = kvh[lev + 1] * tmp2 * rpdel[lev + 1];
    }
    termh[0] = 1.0 / (1.0 + cah[0]);
    zeh[0] = cah[0] * termh[0];
    for (int lev = 1; lev < nlev - 1; ++lev) {
        termh[lev] = 1.0 / (
            1.0 + cah[lev] + cch[lev] * (1.0 - zeh[lev - 1])
        );
        zeh[lev] = cah[lev] * termh[lev];
    }

    zfq[0] = shmx[0] * termh[0];
    for (int lev = 1; lev < nlev - 1; ++lev) {
        zfq[lev] = (shmx[lev] + cch[lev] * zfq[lev - 1])
            * termh[lev];
    }
    const double tmp1d =
        1.0 / (1.0 + cch[nlev - 1] * (1.0 - zeh[nlev - 2]));
    zfq[nlev - 1] = (
        shmx[nlev - 1] + dshbot
        + cch[nlev - 1] * zfq[nlev - 2]
    ) * tmp1d;
    sphu_diffused[nlev - 1] = zfq[nlev - 1];
    for (int lev = nlev - 2; lev >= 0; --lev) {
        sphu_diffused[lev] =
            zfq[lev] + zeh[lev] * sphu_diffused[lev + 1];
    }

    for (int lev = 0; lev < nlev; ++lev) {
        const int center = lev * horizontal_size + horizontal;
        plan_cch[center] = cch[lev];
        plan_zeh[center] = zeh[lev];
        plan_termh[center] = termh[lev];
        plan_rpdel[center] = rpdel[lev];
        double humidity = sphu_diffused[lev];
        if (humidity < 1.0e-12) {
            humidity = 0.0;
        }
        sphu_after_top[center] = humidity;
    }
    for (int edge = 0; edge <= nlev; ++edge) {
        const int target = edge * horizontal_size + horizontal;
        plan_cgs[target] = cgs[edge];
        plan_kvh[target] = kvh[edge];
        plan_potbar[target] = potbar[edge];
    }
    plan_rrho[horizontal] = rrho;
    plan_tmp1[horizontal] = tmp1;
}

template <typename T>
__global__ void prepare_convection_plan(
    const double* cmfmc_bottom,
    const double* dtrain_bottom,
    const double* dqrcu_bottom,
    const double* reevapcn_bottom,
    const double* delp_top,
    T* cmfmc_top,
    T* dtrain_top,
    T* dqrcu_top,
    T* reevapcn_top,
    T* delp,
    T* bmass,
    int nlev,
    int nlat,
    int nlon
) {
    const int target = blockDim.x * blockIdx.x + threadIdx.x;
    const int horizontal_size = nlat * nlon;
    const int center_size = nlev * horizontal_size;
    if (target >= center_size) {
        return;
    }
    const int lev_top = target / horizontal_size;
    const int horizontal = target - lev_top * horizontal_size;
    const int source =
        (nlev - 1 - lev_top) * horizontal_size + horizontal;
    const double pressure = delp_top[target];
    cmfmc_top[target] = static_cast<T>(cmfmc_bottom[source]);
    dtrain_top[target] = static_cast<T>(dtrain_bottom[source]);
    dqrcu_top[target] = static_cast<T>(dqrcu_bottom[source]);
    reevapcn_top[target] = static_cast<T>(reevapcn_bottom[source]);
    delp[target] = static_cast<T>(pressure);
    bmass[target] = static_cast<T>(pressure * (100.0 / 9.80665));
}
