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
 * @date 2026
 * @author Yuichiro Okashita
 */

static int64_t eval_fm_avx2(const int8_t *latent_vector, const uint16_t *f)
{
    int64_t interaction;

#if EVAL_LATENT_VECTOR_DIM >= 16 && EVAL_LATENT_VECTOR_DIM % 16 == 0
    // Native tier: one feature's full row fits (a multiple of) the 256-bit register exactly.
    // This also transparently covers EVAL_LATENT_VECTOR_DIM == 32, 64, ... (NUM_CHUNKS = k/16).
    const int32_t CHUNK_SIZE = 16;
    const int32_t NUM_CHUNKS = EVAL_LATENT_VECTOR_DIM / CHUNK_SIZE;

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
                const __m128i* lv_8 = (const __m128i*)(latent_vector + (LATENT_VECTOR_OFFSET[i] + feature) * EVAL_LATENT_VECTOR_DIM);

                for (int32_t j = 0; j < NUM_CHUNKS; j++) {
                    const __m256i lv_16 = _mm256_cvtepi8_epi16(lv_8[j]);
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

    interaction = (int64_t)sum_sq - (int64_t)sq_sum;

#elif EVAL_LATENT_VECTOR_DIM >= 8 && EVAL_LATENT_VECTOR_DIM < 16 && EVAL_LATENT_VECTOR_DIM % 8 == 0
    // PACK=2: two features' 8-byte rows packed side by side into one 256-bit register.
    // PATTERN_NUM_SYMS[p] (4, or 2 for DIAG_8) is always even, so pairing s,s+1 within a single
    // pattern type's own loop always divides evenly -- no padding ever needed.
    __m256i sum_acc = _mm256_setzero_si256();
    __m256i sq_sum_acc = _mm256_setzero_si256();

    int feature_idx = 0;

    UNROLL_PATTERN_LOOP
    for (int p = 0; p < NUM_PATTERN_TYPES; p++) {
        int num_syms = PATTERN_NUM_SYMS[p];
        if (ACTIVE_FM_PATTERNS[p]) {
            for (int pair = 0; pair < num_syms / 2; pair++) {
                int i0 = feature_idx + 2 * pair, i1 = i0 + 1;
                uint16_t feat0 = f[i0] - FEATURE_OFFSET[i0], feat1 = f[i1] - FEATURE_OFFSET[i1];
                const void *row0 = latent_vector + (LATENT_VECTOR_OFFSET[i0] + feat0) * EVAL_LATENT_VECTOR_DIM;
                const void *row1 = latent_vector + (LATENT_VECTOR_OFFSET[i1] + feat1) * EVAL_LATENT_VECTOR_DIM;

                __m128i r0 = _mm_loadl_epi64((const __m128i*)row0);
                __m128i r1 = _mm_loadl_epi64((const __m128i*)row1);
                __m128i combined = _mm_unpacklo_epi64(r0, r1);        // 16 bytes: row0 | row1
                __m256i lv_16 = _mm256_cvtepi8_epi16(combined);       // widen -> 16 int16 lanes

                sum_acc = _mm256_add_epi16(sum_acc, lv_16);
                sq_sum_acc = _mm256_add_epi32(sq_sum_acc, _mm256_madd_epi16(lv_16, lv_16));
            }
        }
        feature_idx += num_syms;
    }

    // 2-way fold of sum_acc's two 8-lane slots into the true per-dimension total.
    // sq_sum_acc needs no fold: madd_epi16 only ever pairs adjacent lanes within a single
    // feature's contiguous k lanes, so it already gives the exact grand total regardless of packing.
    __m128i sa_lo = _mm256_castsi256_si128(sum_acc);
    __m128i sa_hi = _mm256_extracti128_si256(sum_acc, 1);
    __m128i folded16 = _mm_add_epi16(sa_lo, sa_hi);

    __m256i folded32 = _mm256_cvtepi16_epi32(folded16);
    __m256i sq32 = _mm256_mullo_epi32(folded32, folded32);
    int32_t sum_sq = hsum_epi32(sq32);
    int32_t sq_sum = hsum_epi32(sq_sum_acc);

    interaction = (int64_t)sum_sq - (int64_t)sq_sum;

#else
    #error "eval_fm_avx2: EVAL_LATENT_VECTOR_DIM must be a multiple of 8 and >= 8"
#endif

    return interaction;
}
