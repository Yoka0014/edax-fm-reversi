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
 * @date 2026
 * @author Yuichiro Okashita
 */

static int64_t eval_fm_avx512(const int8_t *latent_vector, const uint16_t *f)
{
    int64_t interaction;

#if EVAL_LATENT_VECTOR_DIM >= 32 && EVAL_LATENT_VECTOR_DIM % 32 == 0
    // Native tier: one feature's full row fits (a multiple of) the 512-bit register exactly.
    const int32_t CHUNK_SIZE = 32;
    const int32_t NUM_CHUNKS = EVAL_LATENT_VECTOR_DIM / CHUNK_SIZE;

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
        if (ACTIVE_FM_PATTERNS[p]) {    // This if-block will be optimized away when compiled with clang -O3.
            for (int s = 0; s < num_syms; s++) {
                int i = feature_idx + s;
                uint16_t feature = f[i] - FEATURE_OFFSET[i];
                const __m256i* lv_8 = (const __m256i*)(latent_vector + (LATENT_VECTOR_OFFSET[i] + feature) * EVAL_LATENT_VECTOR_DIM);

                for (int32_t j = 0; j < NUM_CHUNKS; j++) {
                    const __m512i lv_16 = _mm512_cvtepi8_epi16(lv_8[j]);

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

    interaction = (int64_t)sum_sq - (int64_t)sq_sum;

#elif EVAL_LATENT_VECTOR_DIM >= 16 && EVAL_LATENT_VECTOR_DIM < 32 && EVAL_LATENT_VECTOR_DIM % 16 == 0
    // PACK=2: two features' 16-byte rows packed side by side into one 512-bit register.
    // PATTERN_NUM_SYMS[p] (4, or 2 for DIAG_8) is always even, so pairing s,s+1 within a single
    // pattern type's own loop always divides evenly -- no cross-pattern-type packing needed.
    __m512i sum_acc = _mm512_setzero_si512();
    __m512i sq_sum_acc = _mm512_setzero_si512();

    int feature_idx = 0;

    UNROLL_PATTERN_LOOP
    for (int p = 0; p < NUM_PATTERN_TYPES; p++) {
        int num_syms = PATTERN_NUM_SYMS[p];
        if (ACTIVE_FM_PATTERNS[p]) {
            for (int pair = 0; pair < num_syms / 2; pair++) {
                int i0 = feature_idx + 2 * pair, i1 = i0 + 1;
                uint16_t feat0 = f[i0] - FEATURE_OFFSET[i0], feat1 = f[i1] - FEATURE_OFFSET[i1];
                const __m128i *row0 = (const __m128i*)(latent_vector + (LATENT_VECTOR_OFFSET[i0] + feat0) * EVAL_LATENT_VECTOR_DIM);
                const __m128i *row1 = (const __m128i*)(latent_vector + (LATENT_VECTOR_OFFSET[i1] + feat1) * EVAL_LATENT_VECTOR_DIM);

                __m256i combined = _mm256_set_m128i(_mm_loadu_si128(row1), _mm_loadu_si128(row0)); // 32 bytes: row0|row1
                __m512i lv_16 = _mm512_cvtepi8_epi16(combined);   // 32 int16 lanes: [0:16)=row0, [16:32)=row1

                sum_acc = _mm512_add_epi16(sum_acc, lv_16);
                sq_sum_acc = _mm512_add_epi32(sq_sum_acc, _mm512_madd_epi16(lv_16, lv_16));
            }
        }
        feature_idx += num_syms;
    }

    // 2-way fold of sum_acc's two 16-lane slots into the true per-dimension total.
    // sq_sum_acc needs no fold: madd_epi16 only ever pairs adjacent lanes within a single
    // feature's contiguous k lanes, so it already gives the exact grand total regardless of packing.
    __m256i lo = _mm512_castsi512_si256(sum_acc);
    __m256i hi = _mm512_extracti64x4_epi64(sum_acc, 1);
    __m256i folded16 = _mm256_add_epi16(lo, hi);                 // 16 lanes = true per-dim sum
    __m512i folded32 = _mm512_cvtepi16_epi32(folded16);          // widen 16 lanes -> int32
    __m512i sq32 = _mm512_mullo_epi32(folded32, folded32);

    int32_t sum_sq = hsum_epi32(_mm512_castsi512_si256(sq32)) + hsum_epi32(_mm512_extracti64x4_epi64(sq32, 1));
    int32_t sq_sum = hsum_epi32(_mm512_castsi512_si256(sq_sum_acc)) + hsum_epi32(_mm512_extracti64x4_epi64(sq_sum_acc, 1));

    interaction = (int64_t)sum_sq - (int64_t)sq_sum;

#elif EVAL_LATENT_VECTOR_DIM >= 8 && EVAL_LATENT_VECTOR_DIM < 16 && EVAL_LATENT_VECTOR_DIM % 8 == 0
    // PACK=4: four features' 8-byte rows packed side by side into one 512-bit register.
    // 11 of the 12 pattern types have num_syms==4 -> exactly one packed register op per pattern
    // type (no inner s-loop needed). DIAG_8 (num_syms==2) is zero-padded to fill 4 slots.
    __m512i sum_acc = _mm512_setzero_si512();
    __m512i sq_sum_acc = _mm512_setzero_si512();
    const __m128i zero128 = _mm_setzero_si128();

    int feature_idx = 0;

    UNROLL_PATTERN_LOOP
    for (int p = 0; p < NUM_PATTERN_TYPES; p++) {
        int num_syms = PATTERN_NUM_SYMS[p];
        if (ACTIVE_FM_PATTERNS[p]) {
            __m128i rows[4];
            for (int s = 0; s < 4; s++) {
                if (s < num_syms) {
                    int i = feature_idx + s;
                    uint16_t feature = f[i] - FEATURE_OFFSET[i];
                    const void *row_ptr = latent_vector + (LATENT_VECTOR_OFFSET[i] + feature) * EVAL_LATENT_VECTOR_DIM;
                    rows[s] = _mm_loadl_epi64((const __m128i*)row_ptr);   // 8 real bytes, low 64 bits
                } else {
                    rows[s] = zero128;                                    // DIAG_8 zero-pad
                }
            }
            __m128i ab = _mm_unpacklo_epi64(rows[0], rows[1]);   // 16 bytes: row0 | row1
            __m128i cd = _mm_unpacklo_epi64(rows[2], rows[3]);   // 16 bytes: row2 | row3
            __m256i combined = _mm256_set_m128i(cd, ab);         // 32 bytes: row0|row1|row2|row3
            __m512i lv_16 = _mm512_cvtepi8_epi16(combined);      // widen -> 32 int16 lanes

            sum_acc = _mm512_add_epi16(sum_acc, lv_16);
            sq_sum_acc = _mm512_add_epi32(sq_sum_acc, _mm512_madd_epi16(lv_16, lv_16));
        }
        feature_idx += num_syms;
    }

    // 4-way fold of sum_acc's four 8-lane slots into the true per-dimension total.
    __m256i sa_lo256 = _mm512_castsi512_si256(sum_acc);
    __m256i sa_hi256 = _mm512_extracti64x4_epi64(sum_acc, 1);
    __m128i s0 = _mm256_castsi256_si128(sa_lo256);
    __m128i s1 = _mm256_extracti128_si256(sa_lo256, 1);
    __m128i s2 = _mm256_castsi256_si128(sa_hi256);
    __m128i s3 = _mm256_extracti128_si256(sa_hi256, 1);
    __m128i folded16 = _mm_add_epi16(_mm_add_epi16(s0, s1), _mm_add_epi16(s2, s3));

    __m256i folded32 = _mm256_cvtepi16_epi32(folded16);
    __m256i sq32 = _mm256_mullo_epi32(folded32, folded32);
    int32_t sum_sq = hsum_epi32(sq32);

    __m256i sq_lo = _mm512_castsi512_si256(sq_sum_acc);
    __m256i sq_hi = _mm512_extracti64x4_epi64(sq_sum_acc, 1);
    int32_t sq_sum = hsum_epi32(sq_lo) + hsum_epi32(sq_hi);

    interaction = (int64_t)sum_sq - (int64_t)sq_sum;

#else
    #error "eval_fm_avx512: EVAL_LATENT_VECTOR_DIM must be a multiple of 8 and >= 8"
#endif

    return interaction;
}
