/**
 * @file eval_fm_avx512.c
 *
 * AVX-512 implementation of the FM (Factorization Machine) second-order
 * interaction term used by eval_accumulate(). Bare fragment, included only
 * from eval.c (mirrors the flip_*.c / board.c convention) -- relies on
 * eval.c's file scope for NUM_PATTERN_TYPES, PATTERN_NUM_SYMS,
 * ACTIVE_FM_PATTERNS, LATENT_VECTOR_OFFSET, FEATURE_OFFSET,
 * EVAL_LATENT_VECTOR_DIM, UNROLL_PATTERN_LOOP and hsum_epi32(), all already
 * defined earlier in eval.c.
 *
 * Handles any EVAL_LATENT_VECTOR_DIM >= 8: rows are processed in full
 * 32-byte (one 512-bit register after widening to int16) chunks, with a
 * masked, zero-padded load for a trailing partial chunk when the dimension
 * isn't an exact multiple of 32. The masked-load branch's condition folds to
 * a compile-time constant `false` whenever EVAL_LATENT_VECTOR_DIM % 32 == 0,
 * so the optimizer removes it entirely and this degenerates to the previous
 * fixed-size "native tier" implementation for e.g. the default dim=32.
 *
 * @date 2026
 * @author Yuichiro Okashita
 */

static int64_t eval_fm_avx512(const int8_t *latent_vector, const uint16_t *f)
{
    const int32_t CHUNK_SIZE = 32;
    const int32_t NUM_CHUNKS = (EVAL_LATENT_VECTOR_DIM + CHUNK_SIZE - 1) / CHUNK_SIZE;
    const int32_t REMAINDER = EVAL_LATENT_VECTOR_DIM % CHUNK_SIZE;

    __m512i sum_acc[NUM_CHUNKS];
    __m512i sq_sum_acc[NUM_CHUNKS];

    for (int32_t i = 0; i < NUM_CHUNKS; i++) {
        sum_acc[i] = _mm512_setzero_si512();
        sq_sum_acc[i] = _mm512_setzero_si512();
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
                    __m256i lv_8;
                    if (REMAINDER != 0 && j == NUM_CHUNKS - 1) {
                        // Partial trailing chunk: mask-load zero-pads the missing lanes, which
                        // contribute 0 to both sum_acc and sq_sum_acc below -- no special-casing
                        // needed in the reduction. Dead branch (with REMAINDER a compile-time
                        // constant 0) whenever EVAL_LATENT_VECTOR_DIM is an exact multiple of 32.
                        const __mmask32 mask = (__mmask32)((1u << REMAINDER) - 1);
                        lv_8 = _mm256_maskz_loadu_epi8(mask, row + j * CHUNK_SIZE);
                    } else {
                        lv_8 = _mm256_loadu_si256((const __m256i*)(row + j * CHUNK_SIZE));
                    }
                    const __m512i lv_16 = _mm512_cvtepi8_epi16(lv_8);

                    sum_acc[j] = _mm512_add_epi16(sum_acc[j], lv_16);

                    const __m512i sq_pair = _mm512_madd_epi16(lv_16, lv_16);
                    sq_sum_acc[j] = _mm512_add_epi32(sq_sum_acc[j], sq_pair);
                }
            }
        }
        feature_idx += num_syms;
    }

    int32_t sum_sq = 0;
    int32_t sq_sum = 0;
    for (int32_t i = 0; i < NUM_CHUNKS; i++)
    {
        __m256i sq_sum_acc_lo = _mm512_castsi512_si256(sq_sum_acc[i]);
        __m256i sq_sum_acc_hi = _mm512_extracti64x4_epi64(sq_sum_acc[i], 1);
        sq_sum += hsum_epi32(sq_sum_acc_lo) + hsum_epi32(sq_sum_acc_hi);

        __m256i sum_raw_lo = _mm512_castsi512_si256(sum_acc[i]);
        __m256i sum_raw_hi = _mm512_extracti64x4_epi64(sum_acc[i], 1);

        __m512i sum_val_lo = _mm512_cvtepi16_epi32(sum_raw_lo);
        __m512i sum_sq_lo = _mm512_mullo_epi32(sum_val_lo, sum_val_lo);

        __m512i sum_val_hi = _mm512_cvtepi16_epi32(sum_raw_hi);
        __m512i sum_sq_hi = _mm512_mullo_epi32(sum_val_hi, sum_val_hi);

        __m256i v_lo_256 = _mm512_castsi512_si256(sum_sq_lo);
        __m256i v_hi_256 = _mm512_extracti64x4_epi64(sum_sq_lo, 1);
        sum_sq += hsum_epi32(v_lo_256) + hsum_epi32(v_hi_256);

        v_lo_256 = _mm512_castsi512_si256(sum_sq_hi);
        v_hi_256 = _mm512_extracti64x4_epi64(sum_sq_hi, 1);
        sum_sq += hsum_epi32(v_lo_256) + hsum_epi32(v_hi_256);
    }

    return (int64_t)sum_sq - (int64_t)sq_sum;
}
