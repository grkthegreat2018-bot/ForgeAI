"""probe_llm_memory_math.py — exact memory accounting of LLMs + the capability↔memory coupling.

Part A: decompose WHERE the bytes go (params / optimizer / grads / activations / KV cache)
        and verify each formula numerically.
Part B: WHY better = more memory: loss scales L ~ N^-alpha  ->  memory ~ (loss-excess)^(-1/alpha).
Part C: the time<->space continuum for EXACT decode: cache-all vs recompute vs sqrt-checkpoint.
Part D: why softmax attention resists exact compression (rank inflation + info floor).
"""
import numpy as np

GB = 1 << 30

def gb(b): return b / GB

# ---------------------------------------------------------------- Part A ----
# Toy config ~ 1.2B-class transformer (Jamba-like dims used in this repo).
d_model, n_layers, n_kv_heads, head_dim, vocab = 2048, 24, 8, 64, 32000
n_heads, d_ff = 32, 8192

# --- params (decoder block: attn q,k,v,o + mlp gate/up/down + 2 norms) -------
attn = d_model * (n_heads * head_dim) + 2 * (d_model * n_kv_heads * head_dim) + (n_heads * head_dim) * d_model
mlp = 3 * d_model * d_ff
norms = 2 * d_model
per_layer = attn + mlp + norms
embed = vocab * d_model
N = n_layers * per_layer + embed + d_model  # + final norm
print(f"[A] params N = {N/1e9:.3f}B   (attn {attn/1e6:.0f}M mlp {mlp/1e6:.0f}M per layer)")

BYTES_FP16, BYTES_FP32 = 2, 4
mem_weights_fp16 = N * BYTES_FP16
# AdamW mixed-precision steady state: fp16 weight + fp16 grad + fp32 master + fp32 m + fp32 v
mem_adamw = N * (BYTES_FP16*2 + BYTES_FP32*3)
# activations per layer ~ b*T*d_model * c_act (c_act ~ 10-16 depending on impl, pre-checkpointing)
b, T = 1, 8192
c_act = 12
mem_act = b * T * d_model * n_layers * c_act * BYTES_FP16 // 2  # /2: stored for backward only
# KV cache (inference): 2 (K,V) * T * n_kv_heads * head_dim * bytes * layers * batch
mem_kv = 2 * T * n_kv_heads * head_dim * BYTES_FP16 * n_layers * b
bytes_kv_per_tok = 2 * n_kv_heads * head_dim * BYTES_FP16 * n_layers
print(f"[A] weights fp16      : {gb(mem_weights_fp16):6.2f} GB")
print(f"[A] AdamW train state : {gb(mem_adamw):6.2f} GB  ({mem_adamw/mem_weights_fp16:.1f}x weights)")
print(f"[A] activations b={b} T={T}: {gb(mem_act):6.2f} GB (pre-checkpointing)")
print(f"[A] KV cache T={T}    : {gb(mem_kv):6.2f} GB  ({bytes_kv_per_tok} B/token = {bytes_kv_per_tok/1024:.1f} KiB)")
print(f"[A] KV exceeds weights at T > {mem_weights_fp16/bytes_kv_per_tok:,.0f} tokens")

# ---------------------------------------------------------------- Part B ----
# Chinchilla:  L(N,D) = E + A/N^a + B/D^b   (E=1.69, A=406.4, a=0.34, B=410.7, b=0.28)
E, A_, a_, B_, b_ = 1.69, 406.4, 0.34, 410.7, 0.28
print("\n[B] Chinchilla loss-vs-size coupling:")
for nB in (0.1, 1.0, 10.0, 100.0):
    n = nB * 1e9
    excess = A_ / n ** a_
    print(f"    N={nB:6.1f}B  param-term={excess:.3f}  (+E={E})")
# multiplicative cost to HALVE the param-term  (N2/N1)^(a) = 2 -> N2/N1 = 2^(1/a)
print(f"    halving loss-excess costs x{2**(1/a_):.2f} params (alpha={a_})")
print(f"    => memory ~ (loss-excess)^(-{1/a_:.2f})   near-cubic exponent")

# ---------------------------------------------------------------- Part C ----
# Exact decode time<->space continuum. Per layer, KV bytes/token = bytes_kv_per_tok/n_layers.
# Strategy M(k): store K,V only every k-th position; on miss recompute segment (<=k-1 extra fwds).
# Mean memory ~ T/k blocks ; recompute work per emitted token ~ (k-1)/2 segment fwds amortized.
print("\n[C] exact decode: checkpoint spacing k  (bytes identical, math identical)")
T0 = 8192
for k in (1, 4, 16, 64, 256, 1024):
    kv_bytes = T0 // k * bytes_kv_per_tok
    recompute_frac = (k - 1) / (2 * k)   # fraction of a full re-prefill per token, amortized
    print(f"    k={k:5d}  KV={gb(kv_bytes):7.3f} GB  extra-prefill~{recompute_frac*100:5.1f}%/tok")
print(f"    k=sqrt(T)={int(np.sqrt(T0))}: balanced point, KV={gb(T0//int(np.sqrt(T0))*bytes_kv_per_tok):.3f} GB")

# ---------------------------------------------------------------- Part D ----
# softmax(QK^T) rank inflation: exp of a rank-d_h matrix is typically FULL rank.
rng = np.random.default_rng(0)
t_, dh = 512, 64
Q = rng.standard_normal((t_, dh)); K = rng.standard_normal((t_, dh))
S = Q @ K.T / np.sqrt(dh)
S -= S.max(axis=1, keepdims=True)
P = np.exp(S); P /= P.sum(axis=1, keepdims=True)
sv = np.linalg.svd(P, compute_uv=False)
r99 = int(np.searchsorted(np.cumsum(sv) / sv.sum(), 0.99)) + 1
r_lin = np.linalg.matrix_rank(Q @ K.T)  # pre-softmax rank == head_dim
print(f"\n[D] T={t_} d_h={dh}: rank(QK^T)={r_lin}  rank(softmax)={r_lin and np.linalg.matrix_rank(P)}  r99={r99}")
print("    => exact low-rank replacement of the ATTENTION MATRIX is impossible;")
print("       but O = P V has rank <= d_h, so compression must target K,V themselves.")

# info floor for exact decode: per token per layer must READ all prior keys once
# (q_t . k_j for every j<=t) => Omega(T) reads; total KV footprint is information-
# theoretically T * n_kv * d_h * bytes unless K_t are themselves compressible.
print(f"[D] exact-decode read floor: {2*n_kv_heads*head_dim*BYTES_FP16*n_layers} B/token/layer-group —")
print("    K,V bytes/token is a hard floor UNLESS K_t lives on a compressible manifold (test in probe 2).")
