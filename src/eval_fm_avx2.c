/**
 * @file eval_fm_avx2.c
 *
 * AVX2 implementation of the FM (Factorization Machine) second-order
 * interaction term used by eval_accumulate(). Bare fragment, included only
 * from eval.c (mirrors the flip_*.c / board.c convention) -- relies on
 * eval.c's file scope for NUM_PATTERN_TYPES, PATTERN_NUM_SYMS,
 * ACTIVE_FM_PATTERNS, LATENT_VECTOR_OFFSET, FEATURE_OFFSET,
 * EVAL_LATENT_VECTOR_DIM, UNROLL_PATTERN_LOOP and hsum_epi32(), all already
 * defined earlier in eval.c.
 *
 * Handles any EVAL_LATENT_VECTOR_DIM >= 8: rows are processed in full
 * 16-byte (one 256-bit register after widening to int16) chunks. AVX2 has no
 * byte-granular masked load, so a trailing partial chunk is instead copied
 * into a zero-initialized 16-byte stack buffer (real bytes only) before
 * loading from there. The `REMAINDER` guarding this fallback is a
 * compile-time constant that is 0 whenever EVAL_LATENT_VECTOR_DIM is an
 * exact multiple of 16, so the optimizer removes the whole buffer/memcpy
 * branch in that case, degenerating to the previous fixed-size "native tier"
 * implementation for e.g. the default dim=32.
 *
 * @date 2026
 * @author Yuichiro Okashita
 */

static int64_t eval_fm_avx2(const int8_t *latent_vector, const uint16_t *f)
{
    const int32_t CHUNK_SIZE = 16;
    const int32_t NUM_CHUNKS = (EVAL_LATENT_VECTOR_DIM + CHUNK_SIZE - 1) / CHUNK_SIZE;
    const int32_t REMAINDER = EVAL_LATENT_VECTOR_DIM % CHUNK_SIZE;

    __m256i sum_acc[NUM_CHUNKS];
    __m256i sq_sum_acc[NUM_CHUNKS];

    for (int32_t i = 0; i < NUM_CHUNKS; i++)
    {
        sum_acc[i] = _mm256_setzero_si256();
        sq_sum_acc[i] = _mm256_setzero_si256();
    }

    int feature_idx = 0;

    UNROLL_PATTERN_LOOP
    for (int p = 0; p < NUM_PATTERN_TYPES; p++) {
        int num_syms = PATTERN_NUM_SYMS[p];
        if (ACTIVE_FM_PATTERNS[p]) {
            for (int s = 0; s < num_syms; s++) {
                int i = feature_idx + s;
                uint16_t feature = f[i] - FEATURE_OFFSET[i];
                const int8_t *row = latent_vector + (LATENT_VECTOR_OFFSET[i] + feature) * EVAL_LATENT_VECTOR_DIM;

                for (int32_t j = 0; j < NUM_CHUNKS; j++) {
                    __m128i bytes16;
                    if (REMAINDER != 0 && j == NUM_CHUNKS - 1) {
                        // Partial trailing chunk: zero-pad into a scratch buffer so the missing
                        // lanes contribute 0 to both accumulators below -- no special-casing
                        // needed in the reduction. Dead branch (REMAINDER a compile-time constant
                        // 0) whenever EVAL_LATENT_VECTOR_DIM is an exact multiple of 16.
                        int8_t buf[16] = {0};
                        memcpy(buf, row + j * CHUNK_SIZE, REMAINDER);
                        bytes16 = _mm_loadu_si128((const __m128i*)buf);
                    } else {
                        bytes16 = _mm_loadu_si128((const __m128i*)(row + j * CHUNK_SIZE));
                    }
                    const __m256i lv_16 = _mm256_cvtepi8_epi16(bytes16);
                    sum_acc[j] = _mm256_add_epi16(sum_acc[j], lv_16);

                    const __m256i sq_pair = _mm256_madd_epi16(lv_16, lv_16);
                    sq_sum_acc[j] = _mm256_add_epi32(sq_sum_acc[j], sq_pair);
                }
            }
        }
        feature_idx += num_syms;
    }

    int32_t sum_sq = 0, sq_sum = 0;
    for (int32_t i = 0; i < NUM_CHUNKS; i++) {
        const __m256i sum_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(sum_acc[i]));
        const __m256i sum_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(sum_acc[i], 1));

        const __m256i sum_sq_lo = _mm256_mullo_epi32(sum_lo, sum_lo);
        const __m256i sum_sq_hi = _mm256_mullo_epi32(sum_hi, sum_hi);

        sum_sq += hsum_epi32(sum_sq_lo) + hsum_epi32(sum_sq_hi);
        sq_sum += hsum_epi32(sq_sum_acc[i]);
    }

    return (int64_t)sum_sq - (int64_t)sq_sum;
}
