template <typename T>
__global__ void sample_obsoperator(
    const T* state,
    const double* wet_ps,
    const double* sphu,
    const double* temperature,
    const double* area,
    const double* hyai,
    const double* hybi,
    long long step_time_us,
    int first_entry,
    const int* field_entry,
    const long long* field_tracer,
    const long long* field_to_accumulator,
    const long long* time_start,
    const int* time_count,
    const long long* time_bounds,
    const double* time_weight,
    const long long* horizontal_start,
    const int* horizontal_count,
    const int* horizontal_bounds,
    const signed char* horizontal_type,
    const double* horizontal_weight,
    const double* horizontal_normalization,
    const long long* vertical_start,
    const int* vertical_count,
    const signed char* vertical_type,
    const signed char* vertical_unit,
    const double* vertical_bounds,
    const signed char* vertical_weight_type,
    const double* vertical_weight,
    double* accumulator,
    int* error_flag,
    int field_count,
    int nlev,
    int nlat,
    int nlon,
    int lane_width
) {
    const int field = blockDim.x * blockIdx.x + threadIdx.x;
    if (field >= field_count) {
        return;
    }
    const int entry = field_entry[field];
    if (entry < first_entry) {
        return;
    }
    double temporal = 0.0;
    const long long ts = time_start[entry];
    for (int offset = 0; offset < time_count[entry]; ++offset) {
        const long long index = ts + offset;
        if (time_bounds[2 * index] <= step_time_us &&
            step_time_us < time_bounds[2 * index + 1]) {
            temporal += time_weight[index];
        }
    }
    if (temporal == 0.0) {
        return;
    }

    const long long tracer = field_tracer[field];
    const long long tracer_block = tracer / lane_width;
    const int lane = tracer - tracer_block * lane_width;
    double sample = 0.0;
    const long long hs = horizontal_start[entry];
    const long long vs = vertical_start[entry];

    for (int hoff = 0; hoff < horizontal_count[entry]; ++hoff) {
        const long long h = hs + hoff;
        const int lat_start = horizontal_bounds[4 * h];
        const int lat_end = horizontal_bounds[4 * h + 1];
        const int lon_start = horizontal_bounds[4 * h + 2];
        const int lon_end = horizontal_bounds[4 * h + 3];
        for (int lon = lon_start; lon < lon_end; ++lon) {
            for (int lat = lat_start; lat < lat_end; ++lat) {
                const int horizontal = lat * nlon + lon;
                double horizontal_factor =
                    (horizontal_type[h] == 2 || horizontal_type[h] == 3)
                    ? area[horizontal] : 1.0;
                if (horizontal_type[h] == 1 || horizontal_type[h] == 3) {
                    horizontal_factor /= horizontal_normalization[h];
                } else if (horizontal_type[h] == 4) {
                    horizontal_factor *= horizontal_weight[h];
                }
                const double ps = wet_ps[horizontal];

                for (int voff = 0; voff < vertical_count[entry]; ++voff) {
                    const long long v = vs + voff;
                    const double lower = vertical_bounds[2 * v];
                    const double upper = vertical_bounds[2 * v + 1];
                    int level_start = -1;
                    int level_end = -1;

                    if (vertical_type[v] == 1) {
                        if (vertical_unit[v] == 2) {
                            level_start = static_cast<int>(lower) - 1;
                        } else if (vertical_unit[v] == 0) {
                            int match = -1;
                            for (int candidate = 0; candidate < nlev; ++candidate) {
                                const double edge = hyai[candidate] + hybi[candidate] * ps;
                                if (edge <= lower) {
                                    match = candidate;
                                    break;
                                }
                            }
                            level_start = match < 0 ? nlev - 1 : max(match - 1, 0);
                        } else {
                            double height = 0.0;
                            for (int candidate = 0; candidate < nlev; ++candidate) {
                                const int center = candidate * nlat * nlon + horizontal;
                                const double q = sphu[center];
                                const double avgw =
                                    28.9644 * q / (18.016 * (1.0 - q));
                                const double xh2o = avgw / (1.0 + avgw);
                                const double tv = temperature[center] /
                                    (1.0 - xh2o * (1.0 - 18.016 / 28.9644));
                                const double edge_lower =
                                    hyai[candidate] + hybi[candidate] * ps;
                                const double edge_upper =
                                    hyai[candidate + 1] + hybi[candidate + 1] * ps;
                                height += 287.0 / 9.80665 * tv *
                                    log(edge_lower / edge_upper);
                                if (height >= lower) {
                                    level_start = candidate;
                                    break;
                                }
                            }
                        }
                        if (level_start < 0) {
                            atomicExch(error_flag, 1);
                            continue;
                        }
                        const int source_level = nlev - 1 - level_start;
                        const long long state_index =
                            (((tracer_block * nlev + source_level) * nlat + lat) *
                             nlon + lon) * lane_width + lane;
                        sample += horizontal_factor * vertical_weight[v] *
                            static_cast<double>(state[state_index]);
                        continue;
                    }

                    if (vertical_unit[v] == 2) {
                        level_start = static_cast<int>(lower) - 1;
                        level_end = static_cast<int>(upper) - 1;
                    } else if (vertical_unit[v] == 0) {
                        int start_match = -1;
                        int end_match = -1;
                        for (int candidate = 0; candidate < nlev; ++candidate) {
                            const double edge = hyai[candidate] + hybi[candidate] * ps;
                            if (start_match < 0 && edge <= upper) {
                                start_match = candidate;
                            }
                            if (end_match < 0 && edge <= lower) {
                                end_match = candidate;
                            }
                        }
                        level_start =
                            start_match < 0 ? nlev - 1 : max(start_match - 1, 0);
                        level_end =
                            end_match < 0 ? nlev - 1 : max(end_match - 1, 0);
                    } else {
                        double height = 0.0;
                        for (int candidate = 0; candidate < nlev; ++candidate) {
                            const int center = candidate * nlat * nlon + horizontal;
                            const double q = sphu[center];
                            const double avgw =
                                28.9644 * q / (18.016 * (1.0 - q));
                            const double xh2o = avgw / (1.0 + avgw);
                            const double tv = temperature[center] /
                                (1.0 - xh2o * (1.0 - 18.016 / 28.9644));
                            const double edge_lower =
                                hyai[candidate] + hybi[candidate] * ps;
                            const double edge_upper =
                                hyai[candidate + 1] + hybi[candidate + 1] * ps;
                            height += 287.0 / 9.80665 * tv *
                                log(edge_lower / edge_upper);
                            if (level_start < 0 && height >= lower) {
                                level_start = candidate;
                            }
                            if (level_end < 0 && height >= upper) {
                                level_end = candidate;
                                break;
                            }
                        }
                    }
                    if (level_start < 0 || level_end < 0) {
                        atomicExch(error_flag, 1);
                        continue;
                    }

                    double normalization = 0.0;
                    double operator_sum = 0.0;
                    for (int level = level_start; level <= level_end; ++level) {
                        double weight = 1.0;
                        if (vertical_weight_type[v] == 2 ||
                            vertical_weight_type[v] == 3) {
                            const double edge_lower =
                                hyai[level] + hybi[level] * ps;
                            const double edge_upper =
                                hyai[level + 1] + hybi[level + 1] * ps;
                            weight = edge_lower - edge_upper;
                        }
                        normalization += weight;
                        const int source_level = nlev - 1 - level;
                        const long long state_index =
                            (((tracer_block * nlev + source_level) * nlat + lat) *
                             nlon + lon) * lane_width + lane;
                        operator_sum += weight *
                            static_cast<double>(state[state_index]);
                    }
                    if (vertical_weight_type[v] == 1 ||
                        vertical_weight_type[v] == 3) {
                        operator_sum /= normalization;
                    }
                    sample += horizontal_factor * operator_sum;
                }
            }
        }
    }
    accumulator[field_to_accumulator[field]] += temporal * sample;
}
