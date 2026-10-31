import sys, time
import openvino_genai as ov

MODEL = r".hf_cache\hub\models--OpenVINO--LFM3-1.7B-int4-ov\snapshots\06949a31c3b7273d998f1268a60b88ceccc92197"
device = sys.argv[1] if len(sys.argv) > 1 else "CPU"
p = ov.LLMPipeline(MODEL, device)
c = ov.GenerationConfig()
c.max_new_tokens = 128
prompt = ("Write a Python function `top_k_mask(logits, k)` that returns "
          "a boolean mask True where logits are below the k-th largest "
          "value per row. Reply with only code.")
# warmup
p.generate("def add(a, b):", c)
for i in range(3):
    t = time.time()
    r = p.generate(prompt, c)
    dt = time.time() - t
    print(f"[{device}] 128tok in {dt:.1f}s = {128/dt:.1f} tok/s")
