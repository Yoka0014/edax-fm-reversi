/**
 * @file eval_fm_sse.c
 *
 * SSE2 implementation of the FM (Factorization Machine) second-order
 * interaction term used by eval_accumulate(). Bare fragment, included only
 * from eval.c (mirrors the flip_*.c / board.c convention) -- relies on
 * eval.c's file scope for NUM_PATTERN_TYPES, PATTERN_NUM_SYMS,
 * ACTIVE_FM_PATTERNS, LATENT_VECTOR_OFFSET, FEATURE_OFFSET and
 * EVAL_LATENT_VECTOR_DIM/UNROLL_PATTERN_LOOP, all already defined earlier in
 * eval.c.
 *
 * Deliberately uses only baseline SSE2 intrinsics (matching the existing
 * flip_sse.c/count_last_flip_sse.c convention in this codebase, neither of
 * which use any SSSE3/SSE4.1 intrinsic) -- gating this file on plain
 * defined(__SSE2__) while using an SSE4.1 intrinsic such as
 * _mm_cvtepi8_epi16 would either fail to compile under GCC/clang's bare
 * "-march=x86-64" target (which never defines __SSE4_1__) or bake an
 * illegal instruction into MSVC's maximally-compatible /D__SSE2__-only
 * build (MSVC never defines __SSE4_1__/__SSSE3__ regardless of /arch:).
 *
 * @date 2026
 * @author Yuichiro Okashita
 */

/** Horizontal sum of the 4 int32 lanes of an SSE2 register (no SSSE3 _mm_hadd_epi32 needed). */
static int32_t hsum4_epi32_sse2(__m128i v)
{
    __m128i shuf = _mm_shuffle_epi32(v, _MM_SHUFFLE(2, 3, 0, 1));
    __m128i sums = _mm_add_epi32(v, shuf);
    shuf = _mm_shuffle_epi32(sums, _MM_SHUFFLE(1, 0, 3, 2));
    sums = _mm_add_epi32(sums, shuf);
    return _mm_cvtsi128_si32(sums);
}

static int64_t eval_fm_sse(const int8_t *latent_vector, const uint16_t *f)
{
    int64_t interaction;

#if EVAL_LATENT_VECTOR_DIM >= 8 && EVAL_LATENT_VECTOR_DIM % 8 == 0
    // SSE's native register (128 bits = 8 int16 lanes after widening) is already the smallest
    // requested tier, so there is no packing case here -- one unified chunked/native branch
    // covers k=8, 16, 32, ...
    const int32_t CHUNK_SIZE = 8;
    const int32_t NUM_CHUNKS = EVAL_LATENT_VECTOR_DIM / CHUNK_SIZE;

    __m128i sum_acc[NUM_CHUNKS];
    __m128i sq_sum_acc[NUM_CHUNKS];
    for (int32_t j = 0; j < NUM_CHUNKS; j++) {
        sum_acc[j] = _mm_setzero_si128();
        sq_sum_acc[j] = _mm_setzero_si128();
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
                    __m128i bytes8 = _mm_loadl_epi64((const __m128i*)(row + j * 8));
                    // SSE2-only sign-extend int8->int16 (no _mm_cvtepi8_epi16 / SSE4.1 needed):
                    // unpacklo_epi8(x,x) duplicates each byte into a 16-bit lane with the byte
                    // in the high half, then an arithmetic right shift by 8 sign-extends it.
                    __m128i lv_16 = _mm_srai_epi16(_mm_unpacklo_epi8(bytes8, bytes8), 8);
                    sum_acc[j] = _mm_add_epi16(sum_acc[j], lv_16);
                    sq_sum_acc[j] = _mm_add_epi32(sq_sum_acc[j], _mm_madd_epi16(lv_16, lv_16));
                }
            }
        }
        feature_idx += num_syms;
    }

    int32_t sum_sq = 0, sq_sum = 0;
    for (int32_t j = 0; j < NUM_CHUNKS; j++) {
        // sum_acc[j] already IS the true per-dimension total (PACK=1, no fold needed here).
        // Its magnitude (<= ~46*127 ~= 5842) fits comfortably in int16, so madd_epi16(v,v)
        // directly computes per-dimension squares summed pairwise -- summing all the
        // resulting pairs gives sum_sq exactly, with no separate widen/mullo step needed.
        sum_sq += hsum4_epi32_sse2(_mm_madd_epi16(sum_acc[j], sum_acc[j]));
        sq_sum += hsum4_epi32_sse2(sq_sum_acc[j]);
    }

    interaction = (int64_t)sum_sq - (int64_t)sq_sum;

#else
    #error "eval_fm_sse: EVAL_LATENT_VECTOR_DIM must be a multiple of 8 and >= 8"
#endif

    return interaction;
}
