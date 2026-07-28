/**
 * @file sigma_probe.c
 *
 * Reservoir sampling of probcut nodes, for the eval_sigma re-fit experiment
 * (see the eval-sigma-refit-plan memory / plan file). This module owns the
 * sampling state only; the self-play driver that turns it on/off lives in
 * obftest.c (obf_sigma_probe), and the actual sampling call site is a single
 * hook in midgame.c's search_probcut().
 *
 * @date 2026
 */

#include "sigma_probe.h"
#include "board.h"
#include "search.h"
#include "util.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

bool sigma_probe_active = false;

/** (n_empties, depth, probcut_depth) occurrence counts. depth/probcut_depth can
 * exceed 32 at high search levels, so this is sized generously; out-of-range
 * triples are reported once and skipped rather than crashing a long run. */
static uint64_t sigma_histogram[61][61][61];
static bool sigma_histogram_range_warned = false;

typedef struct SigmaProbeSample {
    Board board;
    int depth;
} SigmaProbeSample;

static SigmaProbeSample *sigma_reservoir = NULL;
static int sigma_reservoir_size = 0;
static int64_t sigma_n_seen = 0;
static Random sigma_reservoir_rng;

void sigma_probe_start(int reservoir_size, uint64_t seed)
{
    free(sigma_reservoir);
    sigma_reservoir_size = reservoir_size;
    sigma_reservoir = (SigmaProbeSample*) malloc(sizeof(SigmaProbeSample) * (size_t) reservoir_size);
    if (sigma_reservoir == NULL) fatal_error("Cannot allocate sigma-probe reservoir\n");
    sigma_n_seen = 0;
    random_seed(&sigma_reservoir_rng, seed);
    memset(sigma_histogram, 0, sizeof sigma_histogram);
    sigma_histogram_range_warned = false;
    sigma_probe_active = true;
}

void sigma_probe_record(Search *search, int depth, int probcut_depth)
{
    const int n_empties = search->n_empties;

    if (n_empties >= 0 && n_empties <= 60 && depth >= 0 && depth <= 60 && probcut_depth >= 0 && probcut_depth <= 60) {
        ++sigma_histogram[n_empties][depth][probcut_depth];
    } else if (!sigma_histogram_range_warned) {
        fprintf(stderr, "sigma-probe: skipping out-of-range triple (e=%d, d=%d, p=%d)\n", n_empties, depth, probcut_depth);
        sigma_histogram_range_warned = true;
    }

    ++sigma_n_seen;
    if (sigma_n_seen <= sigma_reservoir_size) {
        sigma_reservoir[sigma_n_seen - 1].board = search->board;
        sigma_reservoir[sigma_n_seen - 1].depth = depth;
    } else {
        uint64_t j = random_get(&sigma_reservoir_rng) % (uint64_t) sigma_n_seen;
        if (j < (uint64_t) sigma_reservoir_size) {
            sigma_reservoir[j].board = search->board;
            sigma_reservoir[j].depth = depth;
        }
    }
}

void sigma_probe_finish(const char *reservoir_file, const char *histogram_file)
{
    FILE *f;
    int i, n_filled;
    char s[80];

    sigma_probe_active = false;

    n_filled = (int) (sigma_n_seen < sigma_reservoir_size ? sigma_n_seen : sigma_reservoir_size);

    f = fopen(reservoir_file, "w");
    if (f == NULL) fatal_error("Cannot open sigma-probe reservoir file for writing\n");
    for (i = 0; i < n_filled; ++i) {
        board_to_string(&sigma_reservoir[i].board, BLACK, s);
        fprintf(f, "%s;%d\n", s, sigma_reservoir[i].depth);
    }
    fclose(f);

    f = fopen(histogram_file, "w");
    if (f == NULL) fatal_error("Cannot open sigma-probe histogram file for writing\n");
    for (int e = 0; e <= 60; ++e)
        for (int d = 0; d <= 60; ++d)
            for (int p = 0; p <= 60; ++p)
                if (sigma_histogram[e][d][p])
                    fprintf(f, "%d %d %d %" PRIu64 "\n", e, d, p, sigma_histogram[e][d][p]);
    fclose(f);

    free(sigma_reservoir);
    sigma_reservoir = NULL;
    sigma_reservoir_size = 0;
    sigma_n_seen = 0;
}
