/**
 * @file sigma_probe.h
 *
 * Reservoir sampling of probcut nodes, for the eval_sigma re-fit experiment.
 *
 * @date 2026
 */

#ifndef EDAX_SIGMA_PROBE_H
#define EDAX_SIGMA_PROBE_H

#include <stdbool.h>
#include <stdint.h>

struct Search;

/** true while a -sigma-probe run is active; checked by the midgame.c hook. */
extern bool sigma_probe_active;

void sigma_probe_start(int reservoir_size, uint64_t seed);
void sigma_probe_record(struct Search *search, int depth, int probcut_depth);
void sigma_probe_finish(const char *reservoir_file, const char *histogram_file);

#endif
