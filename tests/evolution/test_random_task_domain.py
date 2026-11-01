"""Test the RandomTaskDomain family: math / algorithm / logic.

Runs without a real LLM (dummy solver returns "0" → score 0) and with a
stub solver that returns the correct answer (→ score > 0) to verify the
full generate→prompt→answer→score→RewardGuard pipeline.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from forge.evolution.domain_spec import list_specs, load_spec
from forge.evolution.domains.random_task_domain import RandomTaskDomain
from forge.evolution.simulators import get_simulator, list_simulators


def make_correct_solver(domain):
    """A gen-model stub that reads domain.current_problem and returns the
    real answer as a string (simulating a perfect model)."""

    class _Solver:
        def generate(self, prompt, **gen_params):
            ans = domain.current_problem.get("real_answer", 0)
            if ans == 1.0:
                return "yes"
            if ans == 0.0:
                return "no"
            # Format integers without trailing .0
            if float(ans).is_integer():
                return str(int(ans))
            return str(ans)

    return _Solver()


def main():
    print("=== Registry checks ===")
    sims = list_simulators()
    for s in ["random_math_simulate", "random_algorithm_simulate", "random_logic_simulate"]:
        assert s in sims, f"missing simulator {s}"
        print(f"  simulator registered: {s}")

    specs = list_specs()
    for s in ["random_math", "random_algorithm", "random_logic"]:
        assert s in specs, f"missing spec {s}"
        sp = load_spec(s)
        assert sp.gen_model_type == "llm", f"{s} gen_model_type != llm"
        print(f"  spec loaded: {s} (gen_model_type={sp.gen_model_type}, params={[p.name for p in sp.params]})")

    print("\n=== encode/decode round-trip ===")
    for kind in ["math", "algorithm", "logic"]:
        d = RandomTaskDomain(task_kind=kind, seed=7, include_distractions=False)
        cfg = {"temperature": 0.4, "max_tokens": 80, "top_p": 0.9, "top_k": 40}
        params = d.encode(cfg)
        assert params.shape == (4,), params.shape
        dec = d.decode(params)
        for k in cfg:
            assert abs(dec[k] - cfg[k]) < 1.5, f"{kind} {k}: {dec[k]} vs {cfg[k]}"
        print(f"  {kind}: round-trip OK  params={params.tolist()}  dec={dec}")

    print("\n=== Dummy solver (no gen model) → low score ===")
    for kind in ["math", "algorithm", "logic"]:
        d = RandomTaskDomain(task_kind=kind, seed=3, include_distractions=False)
        assert d._gen_model is None
        cfg = d.decode(torch.tensor([0.0, 0.3, 0.9, 0.4]))
        res = d.evaluate(cfg)
        print(
            f"  {kind}: score={res['score']:.2f}  correct={res['metadata']['correct']}  "
            f"real={res['metadata']['real_answer']}  model='{res['metadata']['model_answer']}'  "
            f"behav={res['behavioral']}"
        )
        assert res["metadata"]["model_answer"] == "0"
        # Dummy answer is "0". Score should be low (no +50 correctness bonus
        # unless real_answer happened to be 0). Cap at the no-correctness max.
        assert res["score"] <= 80.0 + 1e-6

    print("\n=== Correct solver → score > 0 ===")
    for kind in ["math", "algorithm", "logic"]:
        d = RandomTaskDomain(task_kind=kind, seed=11, include_distractions=True, distraction_prob=1.0)
        d.set_gen_model(make_correct_solver(d))
        cfg = d.decode(torch.tensor([0.0, 0.5, 1.0, 0.01]))
        res = d.evaluate(cfg)
        print(
            f"  {kind}: score={res['score']:.2f}  correct={res['metadata']['correct']}  "
            f"focus={res['metadata']['focus_score']}  real={res['metadata']['real_answer']}  "
            f"model='{res['metadata']['model_answer']}'  behav={res['behavioral']}"
        )
        assert res["metadata"]["correct"] == 1.0, f"{kind} expected correct=1.0"
        assert res["metadata"]["focus_score"] == 1.0, f"{kind} expected focus=1.0"
        assert res["score"] > 0, f"{kind} score should be positive when correct"
        # behavioral vector should be (correctness, difficulty)
        assert len(res["behavioral"]) == 2

    print("\n=== Distraction injection present in prompt ===")
    d = RandomTaskDomain(task_kind="math", seed=5, include_distractions=True, distraction_prob=1.0)
    d.set_gen_model(make_correct_solver(d))
    cfg = d.decode(torch.tensor([0.0, 0.5, 1.0, 0.01]))
    res = d.evaluate(cfg)
    prob = res["metadata"]["problem"]
    print(f"  prompt: {prob!r}")
    # Should contain a distraction sentence fragment
    assert any(
        frag in prob for frag in ["sky", "apples", "train", "Pizza", "weather", "cat", "library", "Mountains"]
    ), "no distraction"

    print("\n=== Multiple evaluations produce varied problems ===")
    d = RandomTaskDomain(task_kind="math", seed=99, include_distractions=False)
    d.set_gen_model(make_correct_solver(d))
    cfg = d.decode(torch.tensor([0.0, 0.5, 1.0, 0.01]))
    prompts = set()
    for _ in range(10):
        res = d.evaluate(cfg)
        prompts.add(res["metadata"]["problem"])
    print(f"  unique prompts across 10 evals: {len(prompts)}")
    assert len(prompts) >= 3, "problems should vary across evaluations"

    print("\n=== Simulator direct call (no domain) ===")
    sim = get_simulator("random_math_simulate")
    m = sim({"answer": "7", "time_s": 0.5}, domain=None)
    print(f"  no-domain metrics: {m}")
    assert m["correct"] == 0.0  # real_answer is NaN → not correct

    print("\n=== Domain registry discovery ===")
    from forge.evolution.domains import DOMAINS

    assert "RandomTaskDomain" in DOMAINS, "RandomTaskDomain not auto-discovered"
    print("  RandomTaskDomain in DOMAINS registry: True")
    print(f"  JSON domain classes: {[k for k in DOMAINS if k.startswith('Random') and k != 'RandomTaskDomain']}")

    print("\nALL CHECKS PASSED")


def test_random_task_domain():
    main()


if __name__ == "__main__":
    main()
