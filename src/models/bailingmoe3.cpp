#include "models.h"
#include "llama-memory-recurrent.h"

#include <algorithm>
#include <cmath>
#include <map>

void llama_model_bailingmoe3::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,      hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_MLA,         hparams.n_embd_head_k_mla_impl);
    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_MLA,       hparams.n_embd_head_v_mla_impl);
    ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK,           hparams.n_lora_kv);
    ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,            hparams.n_lora_q, false);
    ml.get_key(LLM_KV_SSM_CONV_KERNEL,                  hparams.ssm_d_conv);
    ml.get_key(LLM_KV_KDA_HEAD_DIM,                     hparams.n_embd_head_kda);
    if (!ml.get_key(LLM_KV_KDA_SAFE_GATE, hparams.kda_safe_gate, false)) {
        hparams.kda_safe_gate = true;
    }
    ml.get_key(LLM_KV_KDA_GATE_LOWER_BOUND,             hparams.kda_gate_lower_bound);
    ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS, hparams.rope_sections, 4, false);
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);
    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,              hparams.n_expert_shared);
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,        hparams.n_layer_dense_lead);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,             hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,              hparams.expert_weights_norm, false);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,               hparams.expert_gating_func);
    ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_EXP,           hparams.swiglu_clamp_exp,   hparams.n_layer_all, false);
    ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_SHEXP,         hparams.swiglu_clamp_shexp, hparams.n_layer_all, false);

    if (hparams.n_ff_shexp == 0) {
        hparams.n_ff_shexp = hparams.n_ff_exp() * std::max(1u, hparams.n_expert_shared);
    }

    GGML_ASSERT(hparams.kda_safe_gate);
    GGML_ASSERT(hparams.kda_gate_lower_bound < 0.0f);

    // the reference alternates H and W pair by pair, so they have one section size, and its T, H and W cover the pairs
    if (hparams.use_mrope() && (hparams.rope_sections[1] != hparams.rope_sections[2] || hparams.rope_sections[3] != 0 ||
            hparams.rope_sections[0] + 2*hparams.rope_sections[1] != (int) hparams.n_rot() / 2)) {
        throw std::runtime_error(format("bailingmoe3: unsupported rope sections [%d, %d, %d, %d] for n_rot %u",
                hparams.rope_sections[0], hparams.rope_sections[1], hparams.rope_sections[2], hparams.rope_sections[3],
                hparams.n_rot()));
    }

    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        hparams.is_recr_impl[il] = hparams.n_head_kv(il) == 0;
    }

    switch (hparams.n_layer()) {
        case 24: type = hparams.n_embd == 1536 && hparams.n_expert == 128 ? LLM_TYPE_7_9B_A1_3B : LLM_TYPE_UNKNOWN; break;
        case 42: type = hparams.n_embd == 2560 && hparams.n_expert == 512 ? LLM_TYPE_124B_A5_1B : LLM_TYPE_UNKNOWN; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_bailingmoe3::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);
    if (output == nullptr) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
    }

    const int64_t head_dim = hparams.n_embd_head_kda;
    const int64_t d_inner = head_dim * n_head;
    const int64_t d_conv = hparams.ssm_d_conv;
    const int64_t kv_lora_rank = hparams.n_lora_kv;
    const int64_t q_lora_rank  = hparams.n_lora_q;
    const int64_t qk_rope_head_dim = hparams.n_rot();
    const int64_t qk_head_dim = hparams.n_embd_head_k_mla();
    const int64_t v_head_dim = hparams.n_embd_head_v_mla();

    const auto nf = nextn_flags(ml);
    const int trunk_flags = nf.trunk;
    const int mtp_flags   = nf.mtp;

    for (int il = 0; il < n_layer; ++il) {
        auto & layer = layers[il];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", il), { n_embd }, trunk_flags);

        if (hparams.is_recr(il)) {
            layer.ssm_q_conv = create_tensor(tn(LLM_TENSOR_SSM_CONV1D_Q, "weight", il), { d_conv, 1, d_inner, 1 }, trunk_flags);
            layer.ssm_k_conv = create_tensor(tn(LLM_TENSOR_SSM_CONV1D_K, "weight", il), { d_conv, 1, d_inner, 1 }, trunk_flags);
            layer.ssm_v_conv = create_tensor(tn(LLM_TENSOR_SSM_CONV1D_V, "weight", il), { d_conv, 1, d_inner, 1 }, trunk_flags);

            create_tensor_qkv(layer, il, n_embd, d_inner, d_inner, d_inner, trunk_flags);
            layer.ssm_f_a = create_tensor(tn(LLM_TENSOR_SSM_F_A, "weight", il), { n_embd, d_inner }, trunk_flags);
            layer.ssm_beta = create_tensor(tn(LLM_TENSOR_SSM_BETA, "weight", il), { n_embd, n_head }, trunk_flags);
            layer.ssm_a = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN, il), { 1, n_head }, trunk_flags);
            layer.ssm_dt_b = create_tensor(tn(LLM_TENSOR_SSM_DT, "bias", il), { d_inner }, trunk_flags);
            layer.ssm_g_a = create_tensor(tn(LLM_TENSOR_SSM_G_A, "weight", il), { n_embd, d_inner }, trunk_flags);
            layer.ssm_o_norm = create_tensor(tn(LLM_TENSOR_SSM_NORM, "weight", il), { head_dim }, trunk_flags);
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), { d_inner, n_embd }, trunk_flags);
        } else {
            if (q_lora_rank > 0) {
                layer.wq_a = create_tensor(tn(LLM_TENSOR_ATTN_Q_A, "weight", il), { n_embd, q_lora_rank }, trunk_flags);
                layer.attn_q_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", il), { q_lora_rank }, trunk_flags);
                layer.wq_b = create_tensor(tn(LLM_TENSOR_ATTN_Q_B, "weight", il), { q_lora_rank, n_head * qk_head_dim }, trunk_flags);
            } else {
                layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", il), { n_embd, n_head * qk_head_dim }, trunk_flags);
            }
            layer.wkv_a_mqa = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_MQA, "weight", il), { n_embd, kv_lora_rank + qk_rope_head_dim }, trunk_flags);
            layer.attn_kv_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", il), { kv_lora_rank }, trunk_flags);
            layer.wk_b = create_tensor(tn(LLM_TENSOR_ATTN_K_B, "weight", il), { qk_head_dim - qk_rope_head_dim, kv_lora_rank, n_head }, trunk_flags);
            layer.wv_b = create_tensor(tn(LLM_TENSOR_ATTN_V_B, "weight", il), { kv_lora_rank, v_head_dim, n_head }, trunk_flags);
            layer.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", il), { n_embd, n_head }, trunk_flags);
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), { n_head * v_head_dim, n_embd }, trunk_flags);
        }

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", il), { n_embd }, trunk_flags);
        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", il), { n_embd, n_ff }, trunk_flags);
            layer.ffn_up = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", il), { n_embd, n_ff }, trunk_flags);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", il), { n_ff, n_embd }, trunk_flags);
        } else {
            layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", il), { n_embd, n_expert }, trunk_flags);
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", il), { n_expert }, trunk_flags);
            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", il), { n_embd, hparams.n_ff_exp(), n_expert }, trunk_flags);
            layer.ffn_up_exps = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS, "weight", il), { n_embd, hparams.n_ff_exp(), n_expert }, trunk_flags);
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il), { hparams.n_ff_exp(), n_embd, n_expert }, trunk_flags);
            layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", il), { n_embd, hparams.n_ff_shexp }, trunk_flags);
            layer.ffn_up_shexp = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP, "weight", il), { n_embd, hparams.n_ff_shexp }, trunk_flags);
            layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", il), { hparams.n_ff_shexp, n_embd }, trunk_flags);
        }
    }

    for (int il = n_layer; il < n_layer_all; ++il) {
        auto & layer = layers[il];
        const int flags = mtp_flags;

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", il), { n_embd }, flags);
        if (q_lora_rank > 0) {
            layer.wq_a = create_tensor(tn(LLM_TENSOR_ATTN_Q_A, "weight", il), { n_embd, q_lora_rank }, flags);
            layer.attn_q_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", il), { q_lora_rank }, flags);
            layer.wq_b = create_tensor(tn(LLM_TENSOR_ATTN_Q_B, "weight", il), { q_lora_rank, n_head * qk_head_dim }, flags);
        } else {
            layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", il), { n_embd, n_head * qk_head_dim }, flags);
        }
        layer.wkv_a_mqa = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_MQA, "weight", il), { n_embd, kv_lora_rank + qk_rope_head_dim }, flags);
        layer.attn_kv_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", il), { kv_lora_rank }, flags);
        layer.wk_b = create_tensor(tn(LLM_TENSOR_ATTN_K_B, "weight", il), { qk_head_dim - qk_rope_head_dim, kv_lora_rank, n_head }, flags);
        layer.wv_b = create_tensor(tn(LLM_TENSOR_ATTN_V_B, "weight", il), { kv_lora_rank, v_head_dim, n_head }, flags);
        layer.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", il), { n_embd, n_head }, flags);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), { n_head * v_head_dim, n_embd }, flags);
        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", il), { n_embd }, flags);
        layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", il), { n_embd, n_expert }, flags);
        layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", il), { n_expert }, flags);
        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", il), { n_embd, hparams.n_ff_exp(), n_expert }, flags);
        layer.ffn_up_exps = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS, "weight", il), { n_embd, hparams.n_ff_exp(), n_expert }, flags);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il), { hparams.n_ff_exp(), n_embd, n_expert }, flags);
        layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", il), { n_embd, hparams.n_ff_shexp }, flags);
        layer.ffn_up_shexp = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP, "weight", il), { n_embd, hparams.n_ff_shexp }, flags);
        layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", il), { hparams.n_ff_shexp, n_embd }, flags);
        layer.nextn.eh_proj = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ, "weight", il), { 2 * n_embd, n_embd }, flags);
        layer.nextn.enorm = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM, "weight", il), { n_embd }, flags);
        layer.nextn.hnorm = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM, "weight", il), { n_embd }, flags);
        layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_LAYER_OUT_NORM, "weight", il), { n_embd }, flags);
    }
}

// The MLA projections write each head's rotary channels pair-interleaved, [x0, y0, x1, y1, ...] (the reference's
// rope_interleave), pair j at frequency j. NORM rope, the rope of the text-only files, rotates those adjacent pairs as
// they are. The VL files use M-RoPE, which rotates the pairs (i, i + n_rot/2) and gives the slots i, from the highest
// frequency, to T, H and W in contiguous sections; the reference's apply_video_rope alternates H and W pair by pair
// over the first 2*s_hw pairs (H on the even ones) and gives T the last s_t. So slot i holds the reference's pair
// bailingmoe3_mrope_pair(i): bailingmoe3_rope_input moves the channels there, and the frequency factors of
// load_tensors give each slot the frequency of its pair. Text positions are equal in T, H and W, so there only the
// frequencies matter.
static int64_t bailingmoe3_mrope_pair(const llama_hparams & hparams, int64_t slot) {
    const int64_t s_t  = hparams.rope_sections[0];
    const int64_t s_hw = hparams.rope_sections[1];
    if (slot < s_t) {
        return 2*s_hw + slot;
    }
    if (slot < s_t + s_hw) {
        return 2*(slot - s_t);
    }
    return 2*(slot - s_t - s_hw) + 1;
}

// [n_rot, heads, tokens] pair-interleaved -> [n_rot, heads, tokens] with x of the pairs in slot order, then y
static ggml_tensor * bailingmoe3_rope_input(ggml_context * ctx, const llama_hparams & hparams, ggml_tensor * x) {
    if (!hparams.use_mrope()) {
        return x;
    }
    const int64_t s_t  = hparams.rope_sections[0];
    const int64_t s_hw = hparams.rope_sections[1];
    const size_t  es   = ggml_element_size(x);

    // the pairs of one section, every stride-th from first, as [pairs, (x, y), heads, tokens]
    auto section = [&](int64_t n_pairs, int64_t first, int64_t stride) {
        ggml_tensor * pairs = ggml_view_4d(ctx, x, 2, n_pairs, x->ne[1], x->ne[2],
                2*stride*es, x->nb[1], x->nb[2], 2*first*es);
        return ggml_permute(ctx, pairs, 1, 0, 2, 3);
    };
    ggml_tensor * slots = ggml_concat(ctx, section(s_t, 2*s_hw, 1), section(s_hw, 0, 2), 0);
    slots = ggml_concat(ctx, slots, section(s_hw, 1, 2), 0);
    return ggml_reshape_3d(ctx, slots, x->ne[0], x->ne[1], x->ne[2]);
}

bool llama_model_bailingmoe3::load_tensors(llama_model_loader & ml) {
    if (!llama_model_base::load_tensors(ml)) {
        return false;
    }
    if (!hparams.use_mrope()) {
        return true;
    }

    // the frequency factors of the M-RoPE slots, the rope_freqs of every attention layer, one tensor per buffer type
    const int64_t n_pairs = hparams.n_rot() / 2;
    std::map<ggml_backend_buffer_type_t, std::pair<ggml_context_ptr, ggml_tensor *>> ctx_map;
    for (uint32_t il = 0; il < hparams.n_layer_all; ++il) {
        auto & layer = layers[il];
        if (layer.wkv_a_mqa == nullptr) {
            continue;
        }
        auto & [ctx, freqs] = ctx_map[derived_buft(layer.wkv_a_mqa)];
        if (!ctx) {
            ggml_init_params params = {
                /*.mem_size   =*/ ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };
            ctx.reset(ggml_init(params));
            if (!ctx) {
                throw std::runtime_error("bailingmoe3: failed to create the context of the M-RoPE frequency factors");
            }
            freqs = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n_pairs);
            ggml_set_name(freqs, "rope_freqs.mrope");
        }
        layer.rope_freqs = freqs;
    }

    // a slot rotates at theta_scale^i / factor, and get_rope_params makes theta_scale 1: the factor is the inverse of
    // the frequency of the slot's pair. A frequency built as theta_scale^i carries the rounding of i products (CPU) or
    // of the pow (Vulkan), a phase drift that grows with the distance between q and k; here the high frequencies sit on
    // high slots, where it is largest, and a frequency from one division drifts less than the plain rope's
    const double freq_base = hparams.rope_freq_base_train;
    std::vector<float> factors(n_pairs);
    for (int64_t i = 0; i < n_pairs; ++i) {
        factors[i] = pow(freq_base, 2.0*bailingmoe3_mrope_pair(hparams, i)/hparams.n_rot());
    }

    for (auto & [buft, entry] : ctx_map) {
        ggml_tensor * freqs = entry.second;
        alloc_derived(std::move(entry.first), buft);
        if (!ml.no_alloc) {
            ggml_backend_tensor_set(freqs, factors.data(), 0, ggml_nbytes(freqs));
        }
    }

    return true;
}

llama_rope_params llama_model_bailingmoe3::get_rope_params(const llama_cparams & cparams, int il) const {
    auto res = llama_model::get_rope_params(cparams, il);

    // the M-RoPE factors of load_tensors carry each slot's whole frequency, from the trained base (a rope base override
    // does not apply); YaRN's ramp would follow the slot order, not the frequency, and the reference has no rope scaling
    if (hparams.use_mrope() && res.factors != nullptr) {
        res.freq_base  = 1.0f;
        res.ext_factor = 0.0f;
    }

    return res;
}

std::unique_ptr<llm_graph_context> llama_model_bailingmoe3::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(params);
    }
    return std::make_unique<graph>(params);
}

ggml_tensor * llama_model_bailingmoe3::graph::build_causal_conv1d(
        llm_graph_input_rs * inp_rs, ggml_tensor * conv_states_all, ggml_tensor * conv_state_all,
        int64_t qkv, ggml_tensor * x, ggml_tensor * proj_w, ggml_tensor * conv_w,
        int64_t d_conv, int64_t head_dim, int64_t n_head, int64_t n_seq_tokens, int64_t n_seqs) {
    const int64_t d_inner = head_dim * n_head;
    const int64_t conv_state_size = (d_conv - 1) * d_inner;
    const int64_t total_state_size = 3 * conv_state_size;

    ggml_tensor * conv_state = ggml_view_3d(ctx0, conv_state_all, d_conv - 1, d_inner, n_seqs,
            (d_conv - 1) * ggml_element_size(conv_state_all),
            total_state_size * ggml_element_size(conv_state_all),
            qkv * conv_state_size * ggml_element_size(conv_state_all));

    ggml_tensor * x_proj = ggml_mul_mat(ctx0, proj_w, x);
    x_proj = ggml_reshape_3d(ctx0, x_proj, d_inner, n_seq_tokens, n_seqs);
    ggml_tensor * conv_x = ggml_concat(ctx0, conv_state, ggml_transpose(ctx0, x_proj), 0);

    build_conv_state_store(inp_rs, conv_states_all, conv_x, d_conv - 1,
            ggml_row_size(conv_states_all->type, qkv * conv_state_size));

    ggml_tensor * conv_weight = ggml_reshape_2d(ctx0, conv_w, d_conv, d_inner);
    ggml_tensor * out = ggml_ssm_conv(ctx0, conv_x, conv_weight);
    out = ggml_silu(ctx0, ggml_reshape_2d(ctx0, out, d_inner, n_tokens));
    return ggml_reshape_4d(ctx0, out, head_dim, n_head, n_seq_tokens, n_seqs);
}

llama_model_bailingmoe3::graph::graph(const llm_graph_params & params) :
    llm_build_delta_net_base(params) {
    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    cb(inpL, "model.input_embed", -1);

    auto * inp = build_inp_mem_hybrid_k();
    auto * inp_rs = inp->get_recr();
    auto * inp_attn = inp->get_attn();

    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    const int64_t n_head = hparams.n_head();
    const int64_t head_dim = hparams.n_embd_head_kda;
    const int64_t d_inner = n_head * head_dim;
    const int64_t d_conv = hparams.ssm_d_conv;
    const int64_t n_seqs = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    const int64_t qk_head_dim = hparams.n_embd_head_k_mla();
    const int64_t v_head_dim = hparams.n_embd_head_v_mla();
    const int64_t qk_rope_head_dim = hparams.n_rot();
    const int64_t qk_nope_head_dim = qk_head_dim - qk_rope_head_dim;
    const int64_t kv_lora_rank = hparams.n_lora_kv;
    const float kq_scale = 1.0f / sqrtf((float) qk_head_dim);

    GGML_ASSERT(n_seqs > 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    for (int il = 0; il < n_layer; ++il) {
        res->t_layer_inp[il] = inpL;

        const auto & layer = model.layers[il];
        ggml_tensor * inpSA = inpL;
        ggml_tensor * cur = build_norm(inpL, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        if (hparams.is_recr(il)) {
            const auto * mctx_cur = inp_rs->mctx;
            ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
            ggml_tensor * conv_state_all = build_rs(inp_rs, conv_states_all, hparams.n_embd_r(), n_seqs);

            ggml_tensor * q = build_causal_conv1d(
                    inp_rs, conv_states_all, conv_state_all, 0, cur, layer.wq, layer.ssm_q_conv,
                    d_conv, head_dim, n_head, n_seq_tokens, n_seqs);
            ggml_tensor * k = build_causal_conv1d(
                    inp_rs, conv_states_all, conv_state_all, 1, cur, layer.wk, layer.ssm_k_conv,
                    d_conv, head_dim, n_head, n_seq_tokens, n_seqs);
            ggml_tensor * v = build_causal_conv1d(
                    inp_rs, conv_states_all, conv_state_all, 2, cur, layer.wv, layer.ssm_v_conv,
                    d_conv, head_dim, n_head, n_seq_tokens, n_seqs);

            ggml_tensor * gate = ggml_mul_mat(ctx0, layer.ssm_f_a, cur);
            gate = ggml_add(ctx0, gate, layer.ssm_dt_b);
            gate = ggml_reshape_3d(ctx0, gate, head_dim, n_head, n_tokens);
            ggml_tensor * a = ggml_reshape_3d(ctx0, layer.ssm_a, 1, n_head, 1);
            gate = ggml_scale(ctx0, ggml_sigmoid(ctx0, ggml_mul(ctx0, gate, a)), hparams.kda_gate_lower_bound);
            gate = ggml_reshape_4d(ctx0, gate, head_dim, n_head, n_seq_tokens, n_seqs);
            cb(gate, "kda_gate", il);

            ggml_tensor * beta = ggml_mul_mat(ctx0, layer.ssm_beta, cur);
            beta = ggml_sigmoid(ctx0, ggml_reshape_4d(ctx0, beta, 1, n_head, n_seq_tokens, n_seqs));

            q = build_gdn_l2_norm(ctx0, q, hparams.f_norm_rms_eps);
            k = build_gdn_l2_norm(ctx0, k, hparams.f_norm_rms_eps);

            ggml_tensor * states_all = mctx_cur->get_s_l(il);
            ggml_tensor * state = build_rs(inp_rs, states_all, hparams.n_embd_s(), n_seqs);
            state = ggml_reshape_4d(ctx0, state, head_dim, head_dim, n_head, n_seqs);

            ggml_tensor * out = ggml_cont(ctx0, build_recurrent_attn(
                    inp_rs, states_all, q, k, v, gate, beta, state, il));

            ggml_tensor * out_gate = ggml_mul_mat(ctx0, layer.ssm_g_a, cur);
            out_gate = ggml_reshape_3d(ctx0, out_gate, head_dim, n_head, n_tokens);
            out = ggml_reshape_3d(ctx0, out, head_dim, n_head, n_tokens);
            out = build_norm(out, layer.ssm_o_norm, nullptr, LLM_NORM_RMS, il);
            out = ggml_mul(ctx0, out, ggml_sigmoid(ctx0, out_gate));
            cur = ggml_mul_mat(ctx0, layer.wo, ggml_cont_2d(ctx0, out, d_inner, n_tokens));
            cb(cur, "kda_out", il);
        } else {
            ggml_tensor * attn_input = cur;
            ggml_tensor * q_all;
            if (layer.wq_a) {
                q_all = ggml_mul_mat(ctx0, layer.wq_a, cur);
                cb(q_all, "q_a", il);
                q_all = build_norm(q_all, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
                cb(q_all, "q_a_norm", il);
                q_all = ggml_mul_mat(ctx0, layer.wq_b, q_all);
                cb(q_all, "q_b", il);
            } else {
                q_all = ggml_mul_mat(ctx0, layer.wq, cur);
            }
            ggml_tensor * q_nope = ggml_view_3d(ctx0, q_all, qk_nope_head_dim, n_head, n_tokens,
                    ggml_row_size(q_all->type, qk_head_dim),
                    ggml_row_size(q_all->type, qk_head_dim) * n_head, 0);
            ggml_tensor * q_pe = ggml_view_3d(ctx0, q_all, qk_rope_head_dim, n_head, n_tokens,
                    ggml_row_size(q_all->type, qk_head_dim),
                    ggml_row_size(q_all->type, qk_head_dim) * n_head,
                    ggml_row_size(q_all->type, qk_nope_head_dim));

            ggml_tensor * kv_all = ggml_mul_mat(ctx0, layer.wkv_a_mqa, cur);
            ggml_tensor * kv = ggml_view_2d(ctx0, kv_all, kv_lora_rank, n_tokens,
                    ggml_row_size(kv_all->type, kv_lora_rank + qk_rope_head_dim), 0);
            ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_all, qk_rope_head_dim, 1, n_tokens,
                    ggml_row_size(kv_all->type, kv_lora_rank + qk_rope_head_dim),
                    ggml_row_size(kv_all->type, kv_lora_rank + qk_rope_head_dim),
                    ggml_row_size(kv_all->type, kv_lora_rank));

            q_pe = build_rope(bailingmoe3_rope_input(ctx0, hparams, q_pe), inp_pos, il);
            k_pe = build_rope(bailingmoe3_rope_input(ctx0, hparams, k_pe), inp_pos, il);
            kv = build_norm(kv, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);

            q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);
            q_nope = ggml_mul_mat(ctx0, layer.wk_b, q_nope);
            q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);

            ggml_tensor * q = ggml_concat(ctx0, q_nope, q_pe, 0);
            kv = ggml_reshape_3d(ctx0, kv, kv_lora_rank, 1, n_tokens);
            ggml_tensor * k = ggml_concat(ctx0, kv, k_pe, 0);

            cur = build_attn(inp_attn, nullptr, nullptr, nullptr,
                    q, k, kv, nullptr, nullptr, layer.wv_b, kq_scale, il);

            ggml_tensor * attn_gate = ggml_mul_mat(ctx0, layer.wqkv_gate, attn_input);
            attn_gate = ggml_sigmoid(ctx0, ggml_reshape_3d(ctx0, attn_gate, 1, n_head, n_tokens));
            cur = ggml_reshape_3d(ctx0, cur, v_head_dim, n_head, n_tokens);
            cur = ggml_mul(ctx0, cur, attn_gate);
            cur = ggml_mul_mat(ctx0, layer.wo, ggml_cont_2d(ctx0, cur, v_head_dim * n_head, n_tokens));
            cb(cur, "mla_out", il);
        }

        if (il == n_layer - 1 && crop_before_nextn(inp_out_ids)) {
            cur = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cur = build_norm(ffn_inp, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            cur = build_ffn(cur,
                    layer.ffn_up, nullptr, nullptr,
                    layer.ffn_gate, nullptr, nullptr,
                    layer.ffn_down, nullptr, nullptr,
                    nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
        } else {
            ggml_tensor * moe = build_moe_ffn(cur,
                    layer.ffn_gate_inp,
                    layer.ffn_up_exps,
                    layer.ffn_gate_exps,
                    layer.ffn_down_exps,
                    layer.ffn_exp_probs_b,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU,
                    hparams.expert_weights_norm,
                    hparams.expert_weights_scale,
                    (llama_expert_gating_func_type) hparams.expert_gating_func,
                    il);
            ggml_tensor * shared = build_ffn(cur,
                    layer.ffn_up_shexp, nullptr, nullptr,
                    layer.ffn_gate_shexp, nullptr, nullptr,
                    layer.ffn_down_shexp, nullptr, nullptr,
                    nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cur = ggml_add(ctx0, moe, shared);
        }

        cur = ggml_add(ctx0, cur, ffn_inp);
        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);
        inpL = cur;
    }

    ggml_tensor * cur = build_norm(inpL, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    if (crop_after_nextn(inp_out_ids)) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;
}

llama_model_bailingmoe3::graph_mtp::graph_mtp(const llm_graph_params & params) :
    llm_graph_context(params) {
    GGML_ASSERT(hparams.n_layer_nextn == 1 && "BailingMoE3 MTP requires one NextN layer");

    const int il = hparams.n_layer() + cparams.nextn_layer_offset;
    GGML_ASSERT(cparams.nextn_layer_offset >= 0 &&
                cparams.nextn_layer_offset < (int) hparams.n_layer_nextn &&
                "nextn_layer_offset out of range");
    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.nextn.eh_proj && "MTP block missing nextn.eh_proj");
    GGML_ASSERT(layer.nextn.enorm && "MTP block missing nextn.enorm");
    GGML_ASSERT(layer.nextn.hnorm && "MTP block missing nextn.hnorm");
    GGML_ASSERT(layer.nextn.shared_head_norm && "MTP block missing final norm");

    const int64_t n_head = hparams.n_head();
    const int64_t qk_head_dim = hparams.n_embd_head_k_mla();
    const int64_t v_head_dim = hparams.n_embd_head_v_mla();
    const int64_t qk_rope_head_dim = hparams.n_rot();
    const int64_t qk_nope_head_dim = qk_head_dim - qk_rope_head_dim;
    const int64_t kv_lora_rank = hparams.n_lora_kv;
    const float kq_scale = 1.0f / sqrtf((float) qk_head_dim);

    auto inp = std::make_unique<llm_graph_input_embd_h>(hparams.n_embd_inp(), hparams.n_embd);
    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);
    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp(), n_tokens);
    ggml_set_input(inp->embd);
    inp->h = build_inp_embd_state(inp.get(), hparams.n_embd);

    ggml_tensor * tok_embd = ubatch.token ? ggml_get_rows(ctx0, model.tok_embd, inp->tokens) : inp->embd;
    ggml_tensor * h_norm = build_norm(inp->h, layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);
    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    ggml_tensor * cur = ggml_mul_mat(ctx0, layer.nextn.eh_proj, ggml_concat(ctx0, e_norm, h_norm, 0));
    cb(cur, "mtp_eh_proj", il);

    res->add_input(std::move(inp));

    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();
    auto * inp_attn = build_attn_inp_k();

    ggml_tensor * inpSA = cur;
    cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
    ggml_tensor * attn_input = cur;

    ggml_tensor * q_all;
    if (layer.wq_a) {
        q_all = ggml_mul_mat(ctx0, layer.wq_a, cur);
        cb(q_all, "q_a", il);
        q_all = build_norm(q_all, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
        cb(q_all, "q_a_norm", il);
        q_all = ggml_mul_mat(ctx0, layer.wq_b, q_all);
        cb(q_all, "q_b", il);
    } else {
        q_all = ggml_mul_mat(ctx0, layer.wq, cur);
    }
    ggml_tensor * q_nope = ggml_view_3d(ctx0, q_all, qk_nope_head_dim, n_head, n_tokens,
            ggml_row_size(q_all->type, qk_head_dim),
            ggml_row_size(q_all->type, qk_head_dim) * n_head, 0);
    ggml_tensor * q_pe = ggml_view_3d(ctx0, q_all, qk_rope_head_dim, n_head, n_tokens,
            ggml_row_size(q_all->type, qk_head_dim),
            ggml_row_size(q_all->type, qk_head_dim) * n_head,
            ggml_row_size(q_all->type, qk_nope_head_dim));

    ggml_tensor * kv_all = ggml_mul_mat(ctx0, layer.wkv_a_mqa, cur);
    ggml_tensor * kv = ggml_view_2d(ctx0, kv_all, kv_lora_rank, n_tokens,
            ggml_row_size(kv_all->type, kv_lora_rank + qk_rope_head_dim), 0);
    ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_all, qk_rope_head_dim, 1, n_tokens,
            ggml_row_size(kv_all->type, kv_lora_rank + qk_rope_head_dim),
            ggml_row_size(kv_all->type, kv_lora_rank + qk_rope_head_dim),
            ggml_row_size(kv_all->type, kv_lora_rank));

    q_pe = build_rope(bailingmoe3_rope_input(ctx0, hparams, q_pe), inp_pos, il);
    k_pe = build_rope(bailingmoe3_rope_input(ctx0, hparams, k_pe), inp_pos, il);
    kv = build_norm(kv, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);

    q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);
    q_nope = ggml_mul_mat(ctx0, layer.wk_b, q_nope);
    q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);

    ggml_tensor * q = ggml_concat(ctx0, q_nope, q_pe, 0);
    kv = ggml_reshape_3d(ctx0, kv, kv_lora_rank, 1, n_tokens);
    ggml_tensor * k = ggml_concat(ctx0, kv, k_pe, 0);

    cur = build_attn(inp_attn, nullptr, nullptr, nullptr,
            q, k, kv, nullptr, nullptr, layer.wv_b, kq_scale, il);

    ggml_tensor * attn_gate = ggml_mul_mat(ctx0, layer.wqkv_gate, attn_input);
    attn_gate = ggml_sigmoid(ctx0, ggml_reshape_3d(ctx0, attn_gate, 1, n_head, n_tokens));
    cur = ggml_reshape_3d(ctx0, cur, v_head_dim, n_head, n_tokens);
    cur = ggml_mul(ctx0, cur, attn_gate);
    cur = ggml_mul_mat(ctx0, layer.wo, ggml_cont_2d(ctx0, cur, v_head_dim * n_head, n_tokens));

    ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
    cur = build_norm(ffn_inp, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);

    ggml_tensor * moe = build_moe_ffn(cur,
            layer.ffn_gate_inp,
            layer.ffn_up_exps,
            layer.ffn_gate_exps,
            layer.ffn_down_exps,
            layer.ffn_exp_probs_b,
            n_expert, n_expert_used,
            LLM_FFN_SILU,
            hparams.expert_weights_norm,
            hparams.expert_weights_scale,
            (llama_expert_gating_func_type) hparams.expert_gating_func,
            il);
    ggml_tensor * shared = build_ffn(cur,
            layer.ffn_up_shexp, nullptr, nullptr,
            layer.ffn_gate_shexp, nullptr, nullptr,
            layer.ffn_down_shexp, nullptr, nullptr,
            nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
    cur = ggml_add(ctx0, moe, shared);
    cur = ggml_add(ctx0, cur, ffn_inp);
    cur = build_norm(cur, layer.nextn.shared_head_norm, nullptr, LLM_NORM_RMS, -1);

    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;
}
