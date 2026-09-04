#pragma once

#include "llama-memory-hybrid.h"

#include <map>
#include <memory>
#include <vector>

//
// llama_memory_hybrid_idx
//

// llama_memory_hybrid plus a third cache with one indexer key per token, for block-sparse attention (qwen4exp QSA)
// the indexer is a side buffer over the attention cells: same size, padding, streams and slots, so cell j is one token in both

class llama_memory_hybrid_idx : public llama_memory_hybrid {
public:
    llama_memory_hybrid_idx(
        const llama_model & model,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr,
                            /* the indexer cache exists only if this is given */
    const layer_filter_cb & filter_idx);

    ~llama_memory_hybrid_idx() = default;

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0)       override;

    //
    // llama_memory_hybrid_idx specific API
    //

    llama_kv_cache * get_mem_idx() const;   // nullptr when the model carries no indexer

    // block-compressed sparse attention (qwen4exp QSA) over the cells of the indexer cache.
    // Blocks cut the position line, not the cell array, so no caller assumes a contiguous layout:
    //   cell_blk  I32 [n_kv, ns]           block each cell belongs to
    //   blk_cells I32 [ratio*n_blocks, ns] cells making up each block
    //   blk_pos   I32 [4*n_blocks*ns]      mrope position rows of each block's first token
    //   bias      F32 [n_kv, n_tokens/ns, ns] -inf where invisible, large where always visible
    // blk_bias asks for the bias per block instead: [n_blocks, n_tokens/ns, ns]
    // the caller then adds the attention mask, the only part of the bias that varies within a block
    // pooled-key cache: with the pooled tables given, blk_cells and blk_pos may be null and the fill
    // resolves the blocks completed since the sequence's watermark instead:
    //   blk_rows    I32 [n_blocks, ns]             cache row each block reads, the dustbin for the rest
    //   dirty_cells I32 [ratio*n_dirty_max, ns]    cells of each block to pool, 0-padded
    //   dirty_pos   I32 [4*n_dirty_max*ns]         mrope position rows of those blocks
    //   dirty_rows  I64 [n_dirty_max*ns]           cache rows to write, dustbin-padded
    void set_input_qsa(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
                       ggml_tensor * bias, const llama_ubatch * ubatch, uint32_t ratio,
                       bool blk_bias,
                       ggml_tensor * blk_rows = nullptr, ggml_tensor * dirty_cells = nullptr,
                       ggml_tensor * dirty_pos = nullptr, ggml_tensor * dirty_rows = nullptr) const;

    // Cache of the indexer's block summary keys (mean-pooled, normalized, roped): one f32 row per
    // (sequence, position block) per QSA layer, written by the graph with set_rows. Only complete
    // blocks are scored and a complete block's members never change, so a row stays valid until a
    // removal or a position change touches its block. Validity is a per-(ratio, sequence) watermark
    // in position blocks: rows below it hold the current content. seq_rm clamps it to the first
    // removed block; seq_cp, seq_add, seq_div, clear and a full state load reset it, and the next
    // ubatch repools the range. Rows at or beyond the watermark are finite but stale, and the
    // fill never lets a scored block read one.

    // pooled key tensor for layer il, or nullptr when the layer has none
    ggml_tensor * get_pooled_k(int32_t il) const;

    // the cache row of (seq_id, position block) at this ratio, and the dustbin row padded writes and unpooled blocks use
    int64_t pooled_row(uint32_t ratio, llama_seq_id seq_id, int64_t blk) const;
    int64_t pooled_row_dustbin(uint32_t ratio) const;

    // position blocks of seq_id (at this ratio) whose rows are valid; mutable through a const context, like the cells
    int64_t & pooled_valid(uint32_t ratio, llama_seq_id seq_id) const;

private:
    // forget seq_id (all of it if seq_id < 0) in every cache at once, so a failed restore cannot leave the caches out of step
    // seq_id < 0 drops the whole context, as the caches themselves do on a failed restore
    void state_drop(llama_seq_id seq_id);

    // the indexer cache holds one key head per layer, so it needs its own hparams:
    // llama_kv_cache keeps a reference to what it is given
    llama_hparams hparams_idx;

    const std::unique_ptr<llama_kv_cache> mem_idx;

    // pooled-key cache storage and watermarks; empty unless the model has an indexer
    ggml_context_ptr        pooled_ctx;
    ggml_backend_buffer_ptr pooled_buf;
    std::map<int32_t, ggml_tensor *> pooled_k;

    uint32_t pooled_n_seq  = 0;
    uint32_t pooled_kv_size = 0;

    mutable std::map<uint32_t, std::vector<int64_t>> pooled_w;   // ratio -> watermark per seq_id

    void pooled_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1);
    void pooled_reset(llama_seq_id seq_id);   // seq_id < 0 resets every sequence
};

class llama_memory_hybrid_idx_context : public llama_memory_hybrid_context {
public:
    using slot_info_vec_t = llama_kv_cache::slot_info_vec_t;

    // used for errors
    explicit llama_memory_hybrid_idx_context(llama_memory_status status);

    // used to create a full-cache context
    explicit llama_memory_hybrid_idx_context(llama_memory_hybrid_idx * mem);

    // used to create an update context
    llama_memory_hybrid_idx_context(
            llama_memory_hybrid_idx * mem,
                      llama_context * lctx,
                               bool   optimize);

    // used to create a batch processing context from a batch
    llama_memory_hybrid_idx_context(
            llama_memory_hybrid_idx * mem,
                    slot_info_vec_t   sinfos_attn,
                    slot_info_vec_t   sinfos_idx,
          std::vector<llama_ubatch>   ubatches);

    ~llama_memory_hybrid_idx_context() = default;

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    //
    // llama_memory_hybrid_idx_context specific API
    //

    // nullptr with no indexer
    const llama_kv_cache_context * get_idx() const;

    // streams in the current slot info, the `ns` of get_k/get_v; 1 if unified
    uint32_t get_n_stream() const;

    void set_input_qsa(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
                       ggml_tensor * bias, const llama_ubatch * ubatch, uint32_t ratio,
                       bool blk_bias,
                       ggml_tensor * blk_rows = nullptr, ggml_tensor * dirty_cells = nullptr,
                       ggml_tensor * dirty_pos = nullptr, ggml_tensor * dirty_rows = nullptr) const;

    // pooled-key cache: the tensor for il, or nullptr when unavailable (no indexer, non-batch context)
    ggml_tensor * get_pooled_k(int32_t il) const;

    // the cache serves a ubatch whose streams each hold a single sequence: rows are keyed per sequence
    bool qsa_pooled_usable(const llama_ubatch & ubatch) const;

    // capacity of the dirty tables for this ubatch, per stream: the blocks completed since the
    // watermark of each stream's sequence; 1 during steady decode, so graph reuse holds
    uint32_t qsa_pooled_n_dirty_max(const llama_ubatch & ubatch, uint32_t ratio) const;

private:
    const llama_memory_hybrid_idx * mem = nullptr;

    // streams per ubatch, read from the slot infos before ctx_idx takes them
    // declared first, so it is initialised while sinfos_idx is still intact
    const std::vector<uint32_t> ns_ubatch;

    // null unless the model has an indexer
    const llama_memory_context_ptr ctx_idx;

    // mirrors the base class's ubatch cursor, which is private there
    size_t i_cur = 0;
};
