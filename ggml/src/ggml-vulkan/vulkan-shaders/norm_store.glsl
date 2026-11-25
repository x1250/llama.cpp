// The end of a norm fused with the rotation of a quantized cache and the K/V store: the workgroup holds its row in
// row_vals[], at most one element per invocation, and every invocation of the workgroup calls these functions.
// The rotation is the fast Walsh-Hadamard transform of fwht.comp and the q8_0 blocks are quantized as copy_to_quant.comp
// does, so that the fused store writes the bytes of the separate ops. The includer defines NORM_STORE_SIZE and, for a
// q8_0 store, NORM_STORE_Q8_0 and data_q[], an array of block_q8_0.

shared float row_vals[NORM_STORE_SIZE];

// the transform of each block of n_rot elements of the row, after its scale: stages of growing span, the lower element
// of each pair becomes their sum and the upper one their difference
void norm_store_fwht(const uint col, const uint ncols, const uint n_rot, const float scale) {
    if (col < ncols) {
        row_vals[col] *= scale;
    }
    barrier();
    for (uint h = 1; h < n_rot; h <<= 1) {
        float v = 0.0f;
        if (col < ncols) {
            const float self  = row_vals[col];
            const float other = row_vals[col ^ h];
            v = (col & h) == 0 ? self + other : other - self;
        }
        barrier();
        if (col < ncols) {
            row_vals[col] = v;
        }
        barrier();
    }
}

#if defined(NORM_STORE_Q8_0)
// block b of the row into data_q[dst_block]
void norm_store_q8_0(const uint b, const uint dst_block) {
    const uint src_idx = b * QUANT_K_Q8_0;

    float amax = 0.0; // absolute max

    [[unroll]] for (int j = 0; j < QUANT_K_Q8_0; j++) {
        const float v = row_vals[src_idx + j];
        amax = max(amax, abs(v));
    }

    const float d = amax / ((1 << 7) - 1);
    const float id = (d != 0.0) ? 1.0/d : 0.0;

    data_q[dst_block].d = float16_t(d);

    [[unroll]] for (int j = 0; j < QUANT_K_Q8_0; ++j) {
        const float x0 = row_vals[src_idx + j]*id;

        data_q[dst_block].qs[j] = int8_t(round(x0));
    }
}
#endif
