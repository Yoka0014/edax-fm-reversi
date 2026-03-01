# Edax FM Reversi

Edax FM Reversi is an extension of Edax that adds **Factorization Machine (FM)** second-order interaction terms to the evaluation function, enabling the engine to capture pairwise feature interactions that the original linear evaluation cannot express.

Its main features are:
- fast bitboard based & multithreaded engine.
- accurate midgame-evaluation function augmented with FM second-order interactions.
- SIMD-accelerated FM computation (AVX-512, AVX2, and scalar fallback).
- opening book learning capability.
- text based rich interface.
- multi-protocol support to connect to graphical interfaces or play on Internet (GGS).
- multi-OS support to run under MS-Windows, Linux and Mac OS X.

## Evaluation function

Let $N$ denote the number of board features, $K$ the latent vector dimension, $v_{i,k} \in \mathbb{Z}$ the quantized (int8) latent vector component for feature $i$ and dimension $k$, and $q_s$ the quantization scale factor.

The evaluation score is:

$$\text{score} = \mathbf{w} \cdot \mathbf{x} + \frac{128}{2 q_s^2} \sum_{k=1}^{K} \left[ \left( \sum_{i=1}^{N} v_{i,k} \right)^2 - \sum_{i=1}^{N} v_{i,k}^2 \right]$$

The bracketed term is the standard FM second-order interaction, computed efficiently as $\text{sum\_sq} - \text{sq\_sum}$. The prefactor $128 / (2 q_s^2)$ dequantizes the result: the true (float) latent vectors are $\tilde{v}_{i,k} = v_{i,k} / q_s$, and the expression above is equivalent to

$$\mathbf{w} \cdot \mathbf{x} + \frac{1}{2} \sum_{k=1}^{K} \left[ \left( \sum_{i=1}^{N} \tilde{v}_{i,k} \right)^2 - \sum_{i=1}^{N} \tilde{v}_{i,k}^2 \right]$$

up to the integer-arithmetic approximation introduced by the $128$ factor.

**Constants:**

| Symbol | Value | Description |
|--------|-------|-------------|
| $N$ | 46 | number of board features |
| $K$ | 32 | latent vector dimension |
| $q_s$ | 128 | quantization scale factor |
| — | 892134 | total number of latent vector rows |

### Evaluation weight file format

The weight file (`eval.dat`) extends the original Edax format by appending the latent vectors immediately after the linear weights:

```
[Edax file header]
[Linear weights: EVAL_N_PLY × packed int16 weight blocks]
[Latent vectors: EVAL_N_LATENT_VECTOR × EVAL_LATENT_VECTOR_DIM bytes, player side]
```

The opponent-side latent vectors are derived at load time by applying the opponent-feature transformation to each row.

## Installation
From [the release section of github](https://github.com/abulmo/edax-reversi/releases), you must 7unzip the evaluation weights (`data/eval.dat`) and place it in the same directory as the executable. The weight file must include the FM latent vectors appended after the linear weights (see format above).
Only 64 bit executables with popcount support are provided.

## Run

### local

```sh
mkdir -p bin
cd src

# Windows (x86-64-v4: AVX-512, recommended for modern Intel/AMD CPUs)
make --makefile=Makefile-clang-windows build ARCH=x86-64-v4

# Windows (x86-64-v3: AVX2, for CPUs without AVX-512)
make --makefile=Makefile-clang-windows build ARCH=x86-64-v3

# e.g. macOS (Apple Silicon)
make pgo-build ARCH=armv8-5a COMP=clang OS=osx
cd ..
./bin/mEdax
```

### docker

```sh
docker build . -t edax
docker run --name "edax" -v "$(pwd)/:/home/edax/" -it edax

cd /home/edax/
mkdir -p bin
cd src
make build ARCH=x86-64-v3 COMP=clang OS=linux

cd ..
# Place the FM-extended eval.dat (with latent vectors) in data/
./bin/lEdax-x64
```

> **Note:** Using `ARCH=x86-64-v4` enables the AVX-512 code path for the FM interaction computation, which processes all 32 latent vector dimensions in a single 512-bit pass. Use `ARCH=x86-64-v3` for the AVX2 path, or omit SIMD flags for the scalar fallback.

## Document

```sh
cd src
doxygen
open ../doc/html/index.html
```
## version 4.6-fm
version 4.6-fm extends version 4.6 with a Factorization Machine evaluation layer:
 - adds 32-dimensional quantized latent vectors for 46 board features.
 - FM second-order interaction computed as `(sum_sq − sq_sum)` over all feature pairs.
 - three SIMD implementations: AVX-512 (one 512-bit pass per feature), AVX2 (two 256-bit passes), and scalar fallback.
 - latent vectors stored in the weight file immediately after the linear weights.
 - opponent-side latent vectors derived at load time via the opponent-feature transform.
 - fixed pointer arithmetic: latent vector row offset is `(LATENT_VECTOR_OFFSET[i] + feature) * EVAL_LATENT_VECTOR_DIM`.
 - fixed opponent latent vector initialization: copies all `EVAL_LATENT_VECTOR_DIM` bytes per row (previously only 1 byte was copied).

## version 4.6
version 4.6 is an evolution of version 4.4 that tried to incorporate changes made by Toshihiko Okuhara in version 4.5.3 and :
 - keep the code encapsulated: I revert many pieces of code from version 4.5.3 with manually inlined code.
 - remove assembly code (intrinsics are good enough)
 - make some changes easily reversible with a macro switch (USE_SIMD, USE_SOLID, etc.)
 - remove buggy code and/or buggy file path.
 - disable code (#if 0) that I found too slow on my cpu.
 - make soft CRC32c behave the same as the hardware CRC32c (version 4.5.3 is buggy here).
 - the code switch from c99 to c17 and use stdatomic.h threads.h (if available) stdalign.h
 - remove bench.c: most of the functions get optimized out and could not be measured.
 - support only 64 bit OSes. 

## makefile
the major change is that the ARCH options are no longer the same, as they are too many possible options to enable avx2, avx512, CRC32c, etc.
Use make -help for a list of options. 


