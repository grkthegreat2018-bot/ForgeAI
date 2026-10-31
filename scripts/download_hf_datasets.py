r"""Download HF datasets for AirMoE expert pack training.

Downloads to D:\windsurf\hf_cache (NOT C drive).
Each dataset is filtered to (prompt, solution) pairs and saved as JSONL
in research/data/expert_training/hf_datasets/<topic>.jsonl

Topics and datasets:
  - coding: MBPP (974 problems), HumanEval (164 problems), CodeAlpaca (20K)
  - python: Python-specific subset of MBPP + CodeSearchNet
  - math: GSM8K (8.8K), MATH (12.5K), MetaMathQA (395K)
  - algorithms: MBPP algorithmic + APPS intro (10K)
  - theory: OpenOrca reasoning subset (50K), FLAN-CoT
  - creativity: CreativeWritingPrompt (1K), StoryWriter
  - tool_use: ToolBench, function-calling datasets
  - token_efficiency: instruction-tuning with concise outputs (Alpaca subset)
  - general: OpenHermes (1M), a general-purpose mix

Usage:
    set HF_TOKEN=<token>
    python scripts/download_hf_datasets.py --topics all
    python scripts/download_hf_datasets.py --topics coding,math --max-samples 5000
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research import paths as _paths

# HF cache lives under the project .devin dir (env vars still win if set).
_HF_CACHE = _paths.as_str(_paths.HF_CACHE_DIR)
os.environ.setdefault("HF_HOME", _HF_CACHE)
os.environ.setdefault("HF_DATASETS_CACHE", f"{_HF_CACHE}/datasets")
os.environ.setdefault("HF_HUB_CACHE", f"{_HF_CACHE}/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", f"{_HF_CACHE}/transformers")

OUTPUT_DIR = _paths.ensure_dir(_paths.HF_DATASETS_DIR)

# Dataset configurations: topic -> list of (hf_id, subset, split, max_samples, extract_fn)
# extract_fn takes a dataset row and returns (prompt, solution, test_cases) or None to skip

def extract_mbpp(row):
    """MBPP: prompt + code + test_imports + test_list."""
    prompt = row.get("text", "") or row.get("prompt", "")
    code = row.get("code", "")
    test_list = row.get("test_list", [])
    if not prompt or not code:
        return None
    tests = [t.strip() for t in test_list if t.strip()]
    return {"prompt": prompt, "solution": code, "test_cases": tests,
            "source": "mbpp", "language": "python"}

def extract_humaneval(row):
    """HumanEval: prompt + canonical_solution."""
    prompt = row.get("prompt", "")
    solution = row.get("canonical_solution", "")
    task_id = row.get("task_id", "")
    if not prompt or not solution:
        return None
    full_prompt = f"Complete the following Python function:\n{prompt}"
    full_solution = prompt + solution
    return {"prompt": full_prompt, "solution": full_solution,
            "test_cases": [], "source": "humaneval", "language": "python",
            "task_id": task_id}

def extract_codealpaca(row):
    """CodeAlpaca: instruction + output."""
    prompt = row.get("instruction", "") or row.get("prompt", "")
    solution = row.get("output", "") or row.get("response", "")
    if not prompt or not solution:
        return None
    return {"prompt": prompt, "solution": solution, "test_cases": [],
            "source": "codealpaca", "language": "python"}

def extract_gsm8k(row):
    """GSM8K: question + answer (with reasoning)."""
    question = row.get("question", "")
    answer = row.get("answer", "")
    if not question or not answer:
        return None
    return {"prompt": question, "solution": answer, "test_cases": [],
            "source": "gsm8k", "language": "math"}

def extract_math(row):
    """MATH dataset: problem + solution."""
    problem = row.get("problem", "")
    solution = row.get("solution", "")
    level = row.get("level", "")
    if not problem or not solution:
        return None
    prompt = f"{problem}\n\nLevel: {level}" if level else problem
    return {"prompt": prompt, "solution": solution, "test_cases": [],
            "source": "math", "language": "math"}

def extract_metamath(row):
    """MetaMathQA: query + response."""
    query = row.get("query", "")
    response = row.get("response", "")
    if not query or not response:
        return None
    return {"prompt": query, "solution": response, "test_cases": [],
            "source": "metamath", "language": "math"}

def extract_openorca(row):
    """OpenOrca: question + response (reasoning subset)."""
    question = row.get("question", "")
    response = row.get("response", "")
    if not question or not response:
        return None
    # Filter for reasoning-type questions
    q_lower = question.lower()
    if any(w in q_lower for w in ["explain", "why", "how", "reason", "analyze", "compare"]):
        return {"prompt": question, "solution": response, "test_cases": [],
                "source": "openorca", "language": "english"}
    return None

def extract_openhermes(row):
    """OpenHermes: conversations format."""
    conv = row.get("conversations", [])
    if not conv or len(conv) < 2:
        return None
    # Find human + gpt turns
    human_turns = [c for c in conv if c.get("from") == "human"]
    gpt_turns = [c for c in conv if c.get("from") == "gpt"]
    if not human_turns or not gpt_turns:
        return None
    prompt = human_turns[0].get("value", "")
    solution = gpt_turns[0].get("value", "")
    if not prompt or not solution:
        return None
    return {"prompt": prompt, "solution": solution, "test_cases": [],
            "source": "openhermes", "language": "english"}

def extract_alpaca(row):
    """Alpaca: instruction + output (filter for concise outputs = token efficiency)."""
    instruction = row.get("instruction", "")
    output = row.get("output", "")
    if not instruction or not output:
        return None
    # For token efficiency: prefer outputs that are concise but complete
    # (under 200 chars but contain actual information)
    if len(output) < 500 and len(output) > 20:
        return {"prompt": instruction, "solution": output, "test_cases": [],
                "source": "alpaca", "language": "english"}
    return None

def extract_toolbench(row):
    """ToolBench / function-calling: prompt + answer with tool use."""
    # Handle prompt as list of messages (chat format)
    prompt_raw = row.get("prompt", "") or row.get("instruction", "") or row.get("query", "")
    if isinstance(prompt_raw, list):
        # Extract user message from chat format
        user_msgs = [m.get("content", "") for m in prompt_raw
                     if isinstance(m, dict) and m.get("role") == "user"]
        prompt = user_msgs[0] if user_msgs else str(prompt_raw)
    elif isinstance(prompt_raw, str):
        prompt = prompt_raw
    else:
        prompt = str(prompt_raw)
    answer = row.get("answer", "") or row.get("response", "") or row.get("completion", "")
    if not prompt or not answer:
        return None
    return {"prompt": prompt, "solution": answer, "test_cases": [],
            "source": "toolbench", "language": "english"}

def extract_creative_writing(row):
    """Creative writing prompts (conversations or instruction format).
    Filters for creative writing keywords in the prompt."""
    # Handle conversations format (list of dicts or JSON string)
    conv = row.get("conversations", [])
    if isinstance(conv, str):
        try:
            conv = json.loads(conv)
        except json.JSONDecodeError:
            conv = []
    if conv and len(conv) >= 2:
        human = next((c for c in conv if c.get("from") == "human"), None)
        gpt = next((c for c in conv if c.get("from") in ("gpt", "model")), None)
        if human and gpt:
            prompt = human.get("value", "")
            story = gpt.get("value", "")
            # Filter for creative writing prompts
            p_lower = prompt.lower()
            creative_kw = ["write a story", "write a poem", "creative", "imagine",
                          "narrative", "fiction", "character", "plot", "dialogue",
                          "describe", "compose", "brainstorm", "tale", "verse"]
            if prompt and story and any(kw in p_lower for kw in creative_kw):
                return {"prompt": prompt, "solution": story, "test_cases": [],
                        "source": "creative_writing", "language": "english"}
    # Fallback: direct fields
    prompt = row.get("prompt", "") or row.get("instruction", "")
    story = row.get("story", "") or row.get("output", "") or row.get("response", "")
    if not prompt or not story:
        return None
    return {"prompt": prompt, "solution": story, "test_cases": [],
            "source": "creative_writing", "language": "english"}

def extract_glaive_toolcall(row):
    """Glaive tool-call dataset (conversations + tools format)."""
    conv = row.get("conversations", [])
    if isinstance(conv, str):
        try:
            conv = json.loads(conv)
        except json.JSONDecodeError:
            conv = []
    if conv and len(conv) >= 2:
        human = next((c for c in conv if c.get("from") == "human"), None)
        gpt = next((c for c in conv if c.get("from") in ("gpt", "model")), None)
        if human and gpt:
            prompt = human.get("value", "")
            answer = gpt.get("value", "")
            tools = row.get("tools", "")
            if tools and isinstance(tools, str):
                prompt = f"Available tools:\n{tools}\n\n{prompt}"
            if prompt and answer:
                return {"prompt": prompt, "solution": answer, "test_cases": [],
                        "source": "glaive_toolcall", "language": "english"}
    return None

def extract_apps_intro(row):
    """APPS introductory level: problem + solutions."""
    question = row.get("question", "")
    solutions = row.get("solutions", "")
    if not question or not solutions:
        return None
    # APPS solutions are JSON string of list
    try:
        sol_list = json.loads(solutions) if isinstance(solutions, str) else solutions
        if sol_list and len(sol_list) > 0:
            solution = sol_list[0] if isinstance(sol_list, list) else str(sol_list)
        else:
            return None
    except (json.JSONDecodeError, TypeError):
        solution = solutions if isinstance(solutions, str) else str(solutions)
    return {"prompt": question, "solution": solution, "test_cases": [],
            "source": "apps_intro", "language": "python"}

def extract_codesearchnet(row):
    """CodeSearchNet: func_code + func_documentation."""
    doc = row.get("func_documentation_string", "")
    code = row.get("func_code_string", "")
    if not doc or not code:
        return None
    return {"prompt": doc, "solution": code, "test_cases": [],
            "source": "codesearchnet", "language": "python"}

# Topic -> dataset configs
TOPIC_DATASETS = {
    "coding": [
        ("google-research-datasets/mbpp", None, "train", 5000, extract_mbpp),
        ("openai/openai_humaneval", None, "test", 5000, extract_humaneval),
        ("sahil2801/CodeAlpaca-20k", None, "train", 10000, extract_codealpaca),
    ],
    "python": [
        ("google-research-datasets/mbpp", None, "train", 5000, extract_mbpp),
        ("sahil2801/CodeAlpaca-20k", None, "train", 10000, extract_codealpaca),
        ("espejelomar/code_search_net_python_10000_examples", None, "train", 5000, extract_codesearchnet),
    ],
    "math": [
        ("openai/gsm8k", "main", "train", 5000, extract_gsm8k),
        ("qwedsacf/competition_math", None, "train", 10000, extract_math),
        ("meta-math/MetaMathQA", None, "train", 20000, extract_metamath),
    ],
    "algorithms": [
        ("google-research-datasets/mbpp", None, "train", 5000, extract_mbpp),
        ("ziwenyd/leetcode-standalone", None, "train", 5000, extract_mbpp),
    ],
    "theory": [
        ("Open-Orca/OpenOrca", None, "train", 30000, extract_openorca),
    ],
    "creativity": [
        ("teknium/OpenHermes-2.5", None, "train", 5000, extract_creative_writing),
    ],
    "tool_use": [
        ("llamafactory/glaive_toolcall_en", None, "train", 5000, extract_glaive_toolcall),
        ("pranavvmurthy26/synthetic-financial-tool-calling-grpo-rlvr-1k", None, "train", 5000, extract_toolbench),
    ],
    "token_efficiency": [
        ("tatsu-lab/alpaca", None, "train", 15000, extract_alpaca),
    ],
    "general": [
        ("teknium/OpenHermes-2.5", None, "train", 30000, extract_openhermes),
    ],
}


def download_topic(topic: str, max_samples: int = 5000):
    """Download all datasets for a topic and save as JSONL."""
    from datasets import load_dataset

    configs = TOPIC_DATASETS.get(topic, [])
    if not configs:
        print(f"  [SKIP] Unknown topic: {topic}")
        return 0

    output_file = OUTPUT_DIR / f"{topic}.jsonl"
    total = 0

    with open(output_file, "w", encoding="utf-8") as f:
        for hf_id, subset, split, default_max, extract_fn in configs:
            n_target = min(max_samples, default_max)
            print(f"  [{topic}] Downloading {hf_id} ({subset or 'default'}) "
                  f"split={split}, max={n_target}...")

            try:
                if subset:
                    ds = load_dataset(hf_id, subset, split=split,
                                      token=os.environ.get("HF_TOKEN"))
                else:
                    ds = load_dataset(hf_id, split=split,
                                      token=os.environ.get("HF_TOKEN"))

                n_written = 0
                for row in ds:
                    if n_written >= n_target:
                        break
                    result = extract_fn(row)
                    if result is not None:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        n_written += 1

                total += n_written
                print(f"    Extracted {n_written} samples from {hf_id}")

            except Exception as e:
                print(f"    [ERROR] {hf_id}: {e}")
                # Try alternative dataset names
                if "Repository Not Found" in str(e) or "401" in str(e) or "403" in str(e):
                    print(f"    Trying alternative datasets for {topic}...")
                continue

    print(f"  [{topic}] Total: {total} samples -> {output_file}")
    return total


def main():
    parser = argparse.ArgumentParser(description="Download HF datasets for expert packs")
    parser.add_argument("--topics", type=str, default="all",
                        help="Comma-separated topics or 'all'")
    parser.add_argument("--max-samples", type=int, default=5000,
                        help="Max samples per dataset")
    args = parser.parse_args()

    if not os.environ.get("HF_TOKEN"):
        print("WARNING: HF_TOKEN not set. Public datasets will work, "
              "gated ones will fail.")

    if args.topics == "all":
        topics = list(TOPIC_DATASETS.keys())
    else:
        topics = [t.strip() for t in args.topics.split(",")]

    print(f"\n{'='*60}")
    print(f"Downloading HF datasets to {OUTPUT_DIR}")
    print(f"HF cache: {os.environ['HF_HOME']}")
    print(f"Topics: {', '.join(topics)}")
    print(f"Max samples per dataset: {args.max_samples}")
    print(f"{'='*60}\n")

    grand_total = 0
    for topic in topics:
        print(f"\n--- {topic} ---")
        n = download_topic(topic, args.max_samples)
        grand_total += n

    print(f"\n{'='*60}")
    print(f"DONE: {grand_total} total samples across {len(topics)} topics")
    print(f"Output: {OUTPUT_DIR}/")
    for topic in topics:
        f = OUTPUT_DIR / f"{topic}.jsonl"
        if f.exists():
            size_mb = f.stat().st_size / 1e6
            print(f"  {topic}.jsonl: {size_mb:.1f} MB")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
