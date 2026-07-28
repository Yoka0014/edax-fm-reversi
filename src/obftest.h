/**
 * @file obftest.h
 *
 * @brief Problem solver.
 *
 * @date 1998 - 2024
 * @author Richard Delorme
 * @version 4.6
 */

#ifndef EDAX_OPDTEST_H
#define EDAX_OPDTEST_H

#include <stdint.h>

struct Search;

void obf_test(struct Search*, const char*, const char*);
void script_to_obf(struct Search*, const char*, const char*);
void obf_filter(const char*, const char *);
void obf_speed(struct Search*, const int, int, int, int);
void obf_sigma_probe(struct Search*, int n_games, int opening_plies, int reservoir_size,
                      uint64_t seed, const char *reservoir_file, const char *histogram_file);
void obf_sigma_scan(struct Search*, const char *input_file, const char *output_file, int dmax);

#endif /* EDAX_OPDTEST_H */

