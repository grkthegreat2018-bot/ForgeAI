"""Test the trained LoRA adapter on novel tool-use prompts."""
import os
import sys

os.environ["FORGE_NO_COMPILE"] = "1"
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)
os.chdir(_project_root)

import torch

from forge.engine.forge_engine import ForgeEngine
from forge.self_play.discovery.qwen_adapter import render_messages_for_config

# Load V2 + ForgeHybrid + OutRo + ForgeQuant
print("Loading ForgeLM V2 + ForgeHybrid + OutRo + ForgeQuant...")
engine = ForgeEngine.from_checkpoint(
    checkpoint="research/checkpoints/ForgeLM_V2.safetensors",
    config_name="forgelm_v2",
    config_overrides={"use_forge_hybrid": True, "use_outro": True},
    auto_activate=False,
)
engine.activate(quantize="forge_quant", kv_cache="rotorquant")
print(f"  Engine loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# Load the trained LoRA adapter
print("Loading LoRA adapter...")
n_loaded = engine.load_lora("research/checkpoints/forgelm_v2_tooluse.lora.safetensors")
print(f"  LoRA tensors loaded: {n_loaded}")

# Test prompts — novel (not in Glaive training data)
test_prompts = [
    # 1. Tool call: web search (novel query)
    [
        {"role": "system", "content": "You are a helpful assistant with access to tools. Use them when needed."},
        {"role": "user", "content": "What's the latest news about quantum computing?"},
    ],
    # 2. Tool call: time tool (novel)
    [
        {"role": "system", "content": "You are a helpful assistant with access to tools."},
        {"role": "user", "content": "What time is it right now?"},
    ],
    # 3. No tool needed — general knowledge
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain how a transformer attention mechanism works in 2 sentences."},
    ],
    # 4. Tool call: memory recall (novel)
    [
        {"role": "system", "content": "You are a helpful assistant with access to tools."},
        {"role": "user", "content": "Do you remember what we discussed about ForgeAI last week?"},
    ],
]

for i, messages in enumerate(test_prompts):
    print(f"\n=== Test {i+1} ===")
    print(f"User: {messages[-1]['content']}")
    rendered = render_messages_for_config(messages, config_name="forgelm_v2")
    output = engine.generate(
        rendered, max_new_tokens=200, temperature=0.7, top_p=0.9,
        repetition_penalty=1.1,
    )
    print(f"Assistant: {output.strip()}")
    print(f"  (VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB)")
