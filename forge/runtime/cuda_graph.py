"""CUDA Graph inference for eliminating kernel launch overhead.

CUDA graphs capture a sequence of GPU operations into a graph that can be
replayed with minimal CPU overhead. For autoregressive generation, this
gives 30-50% speedup on small models (which are kernel-launch-bound, not
compute-bound).

Key constraint: CUDA graphs require fixed tensor shapes. We handle this by:
1. Static batch size and sequence length during capture
2. Padding inputs to fixed size
3. Using input/output copy for dynamic content

Usage:
    from forge.runtime.cuda_graph import CudaGraphRunner

    runner = CudaGraphRunner(model, batch_size=1, seq_len=1, device="cuda")
    runner.capture()  # capture the graph
    output = runner.run(input_ids)  # replay (fast)
"""
import torch


class CudaGraphRunner:
    """CUDA Graph capture and replay for fast inference.

    Captures a single forward pass into a CUDA graph. Replays it with
    new inputs by copying into pre-allocated input buffers.

    Args:
        model: the LLM
        batch_size: fixed batch size for the graph
        seq_len: fixed sequence length (1 for decode, T for prefill)
        device: must be cuda
        use_cache: if True, capture with KV cache (for autoregressive gen)
    """

    def __init__(self, model, batch_size=1, seq_len=1, device="cuda",
                 use_cache=True):
        self.model = model
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = torch.device(device)
        self.use_cache = use_cache
        self.captured = False

        # Pre-allocate input/output buffers.
        self.static_input_ids = torch.zeros(
            batch_size, seq_len, dtype=torch.long, device=self.device
        )
        self.static_output = None  # will be set after first forward
        self.graph = None

    def _forward(self):
        """Forward pass with static buffers."""
        if self.use_cache:
            out = self.model(self.static_input_ids, use_cache=True)
        else:
            out = self.model(self.static_input_ids)
        return out

    def capture(self):
        """Capture the CUDA graph.

        Must be called on CUDA device. The model should be in eval mode.
        """
        if self.device.type != "cuda":
            print("  [CudaGraph] WARNING: CUDA graphs require CUDA device, skipping")
            return False

        self.model.eval()

        # Warmup (required before capture).
        with torch.no_grad():
            for _ in range(3):
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    out = self._forward()
                torch.cuda.current_stream().wait_stream(s)

        # Capture.
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            out = self._forward()

        # Store output reference.
        if isinstance(out, tuple):
            self.static_output = out[0]
        else:
            self.static_output = out

        self.captured = True
        print(f"  [CudaGraph] captured: batch={self.batch_size}, seq_len={self.seq_len}")
        return True

    def run(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run the captured graph with new input.

        Args:
            input_ids: (B, T) must match batch_size and seq_len

        Returns:
            output tensor (same shape as captured)
        """
        if not self.captured:
            # Fallback: regular forward.
            with torch.no_grad():
                out = self._forward()
                return out[0] if isinstance(out, tuple) else out

        # Copy input into static buffer.
        self.static_input_ids.copy_(input_ids)

        # Replay graph.
        self.graph.replay()

        # Return output (clone to avoid mutation issues).
        return self.static_output.clone()


class CudaGraphGenerator:
    """Autoregressive generation with CUDA graph acceleration.

    Uses two graphs:
    1. Prefill graph: processes the prompt (variable length → pad to fixed)
    2. Decode graph: generates one token at a time (fixed shape, fast)

    Args:
        model: the LLM
        max_batch_size: max concurrent sequences
        max_prompt_len: max prompt length (for prefill padding)
        device: cuda
    """

    def __init__(self, model, max_batch_size=1, max_prompt_len=512, device="cuda"):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_prompt_len = max_prompt_len
        self.device = torch.device(device)

        # We'll capture the decode graph (single token, with KV cache).
        self.decode_runner = None
        self.captured = False

    def capture(self):
        """Capture the decode step graph."""
        if self.device.type != "cuda":
            return False

        # Capture decode: single token forward with KV cache.
        self.decode_runner = CudaGraphRunner(
            self.model, batch_size=self.max_batch_size, seq_len=1,
            device=self.device, use_cache=True
        )
        return self.decode_runner.capture()

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 100,
                 temperature: float = 0.0, eos_token_id: int | None = None) -> torch.Tensor:
        """Generate tokens with CUDA graph acceleration.

        Args:
            input_ids: (1, T) prompt
            max_new_tokens: max tokens to generate
            temperature: 0 for greedy
            eos_token_id: stop token

        Returns:
            (1, T + generated) token ids
        """
        input_ids = input_ids.to(self.device)
        B, _T = input_ids.shape

        # Prefill (not graphed — variable length).
        out = self.model(input_ids, use_cache=True)
        logits = out[0] if isinstance(out, tuple) else out
        past_kv = out[2] if isinstance(out, tuple) and len(out) > 2 else None
        if past_kv is None and isinstance(out, tuple) and len(out) >= 2:
            past_kv = out[1]

        # Generate first token.
        if temperature == 0:
            next_token = logits[:, -1, :].argmax(-1, keepdim=True)
        else:
            probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        generated = [next_token]

        # Decode loop (graphed if captured).
        for _ in range(max_new_tokens - 1):
            if eos_token_id and next_token.item() == eos_token_id:
                break

            if self.captured and self.decode_runner:
                # Use graphed decode.
                # Need to pad to max_batch_size.
                if self.max_batch_size > B:
                    pad = torch.full(
                        (self.max_batch_size - B, 1), 0,
                        dtype=torch.long, device=self.device
                    )
                    decode_input = torch.cat([next_token, pad], dim=0)
                else:
                    decode_input = next_token

                out = self.decode_runner.run(decode_input)
                logits = out[:B]  # slice back to actual batch
            else:
                # Regular decode.
                out = self.model(next_token, past_key_values=past_kv, use_cache=True)
                logits = out[0] if isinstance(out, tuple) else out
                past_kv = out[2] if isinstance(out, tuple) and len(out) > 2 else past_kv

            if temperature == 0:
                next_token = logits[:, -1, :].argmax(-1, keepdim=True)
            else:
                probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            generated.append(next_token)

        # Concatenate.
        all_generated = torch.cat(generated, dim=1)
        return torch.cat([input_ids, all_generated], dim=1)


def benchmark_cuda_graph(model, input_ids, n_warmup=5, n_runs=20, device="cuda"):
    """Benchmark regular vs CUDA graph inference.

    Args:
        model: the LLM
        input_ids: (B, T) test input
        n_warmup: warmup iterations
        n_runs: benchmark iterations

    Returns:
        dict with regular_tps, graph_tps, speedup
    """
    import time

    if device.type != "cuda":
        return {"regular_tps": 0, "graph_tps": 0, "speedup": 1.0}

    model.eval()
    B, T = input_ids.shape

    # Warmup.
    with torch.no_grad():
        for _ in range(n_warmup):
            model(input_ids)
            if device.type == "cuda":
                torch.cuda.synchronize()

    # Regular inference.
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            model(input_ids)
    torch.cuda.synchronize()
    regular_time = (time.time() - t0) / n_runs

    # CUDA graph.
    runner = CudaGraphRunner(model, batch_size=B, seq_len=T, device=device)
    runner.capture()

    # Warmup graph.
    for _ in range(n_warmup):
        runner.run(input_ids)
    torch.cuda.synchronize()

    # Graph inference.
    t0 = time.time()
    for _ in range(n_runs):
        runner.run(input_ids)
    torch.cuda.synchronize()
    graph_time = (time.time() - t0) / n_runs

    speedup = regular_time / graph_time
    print(f"  [CudaGraph] regular: {regular_time*1000:.2f}ms | "
          f"graph: {graph_time*1000:.2f}ms | {speedup:.2f}x speedup")

    return {"regular_ms": regular_time * 1000, "graph_ms": graph_time * 1000,
            "speedup": speedup}
