#ifdef MUL_MAT_ID
shared u16vec2 row_ids[BN];
uint _ne1;

#define MAT_MAT_ID_FUSION_FLAGS_ROW_SCALE 0x1

// MUL_MAT_ID_MUL: each row of the product (expert slot row_idx.x, token row_idx.y) times its value in a
// [1, n_expert_used, n_tokens] tensor, the expert scales or the router weights
float row_scale(const u16vec2 row_idx) {
    return (p.fusion_flags & MAT_MAT_ID_FUSION_FLAGS_ROW_SCALE) != 0 ? data_row_scale[row_idx.x + row_idx.y * p.nei0] : 1.0f;
}

void load_row_ids(uint row_begin) {
    for (uint i = gl_LocalInvocationIndex; i < _ne1; i += BLOCK_SIZE) {
        const uint packed_row_id = data_row_map[p.row_ids_offset + row_begin + i];
        row_ids[i] = u16vec2(packed_row_id & 0xffffu, packed_row_id >> 16);
    }
    barrier();
}
#endif // MUL_MAT_ID
