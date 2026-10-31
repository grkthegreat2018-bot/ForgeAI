"""Curriculum learning and data augmentation for LLM pretraining.

Based on:
  1. "Beyond Random Sampling: Efficient Language Model Pretraining via
     Curriculum Learning" (EACL 2026): 18-45% fewer steps, 3.5% sustained
     improvement. Best difficulty signals: compression ratio, lexical
     diversity (MTLD), readability (Flesch Reading Ease).
  2. "Demystifying Training-Time Augmentation for Data-Constrained Language
     Model Pretraining" (arXiv 2606.16246): three augmentation categories:
     token-level noise, sequence permutations, target offset prediction.
  3. "SYNPRO: Generating Pretraining Tokens from Organic Data for Data-Bound
     Scaling" (arXiv 2605.17849): rephrasing + reformat operations,
     RL-optimized generators, 3.7-5.2× effective tokens.

For our self-play + SFT training:
  - Curriculum: order training data easy→hard (18-45% faster convergence)
  - Augmentation: regularize against overfitting in multi-epoch training
  - SYNPRO: generate diverse rephrasings of training data (data-bound regime)
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass

import torch

# ─── Curriculum Learning ───

@dataclass
class DifficultyScore:
    """Difficulty score for a training sample."""
    compression_ratio: float = 0.0  # compressed/original size
    lexical_diversity: float = 0.0  # MTLD (Measure of Textual Lexical Diversity)
    readability: float = 0.0        # Flesch Reading Ease
    combined: float = 0.0           # weighted combination


def compute_difficulty(text: str) -> DifficultyScore:
    """Compute difficulty score for a text sample.

    Based on EACL 2026 findings: compression ratio, lexical diversity (MTLD),
    and readability (Flesch Reading Ease) are the most effective signals.
    """
    # Compression ratio: compressed size / original size
    import zlib
    compressed = len(zlib.compress(text.encode('utf-8')))
    original = len(text.encode('utf-8'))
    comp_ratio = compressed / max(original, 1)

    # Lexical diversity: MTLD (Measure of Textual Lexical Diversity)
    words = text.split()
    mtld = compute_mtld(words)

    # Readability: Flesch Reading Ease
    flesch = compute_flesch(text)

    # Combined score (higher = harder)
    # Compression: higher ratio = more repetitive = easier
    # MTLD: higher = more diverse = harder
    # Flesch: lower = harder
    combined = (1 - comp_ratio) * 0.3 + mtld * 0.3 + (100 - flesch) / 100 * 0.4

    return DifficultyScore(
        compression_ratio=comp_ratio,
        lexical_diversity=mtld,
        readability=flesch,
        combined=combined,
    )


def compute_mtld(words: list[str]) -> float:
    """Compute MTLD (Measure of Textual Lexical Diversity).

    MTLD: average length of sequential word strings before a type/token
    ratio (TTR) drops below a threshold (typically 0.72).
    """
    if not words:
        return 0.0

    threshold = 0.72
    factors = 0.0
    factor_lengths = []

    # Forward pass
    types = set()
    for i, word in enumerate(words):
        types.add(word.lower())
        ttr = len(types) / (i + 1)
        if ttr < threshold:
            factors += 1
            factor_lengths.append(i + 1)
            types = set()

    # Handle remainder
    if types:
        # Simplified: add partial factor
        factors += (len(types) / len(words)) * 0.5

    if factors == 0:
        return 1.0  # all unique

    avg_length = len(words) / factors
    # Normalize to 0-1
    return min(1.0, avg_length / 100)


def compute_flesch(text: str) -> float:
    """Compute Flesch Reading Ease score.

    0-30: very difficult (graduate level)
    30-50: difficult
    50-60: fairly difficult
    60-70: standard
    70-80: fairly easy
    80-100: easy
    """
    sentences = len(re.findall(r'[.!?]+', text)) or 1
    words = len(text.split()) or 1
    # Count syllables (simplified: count vowel groups)
    syllables = sum(count_syllables(word) for word in text.split()) or 1

    flesch = 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)
    return max(0, min(100, flesch))


def count_syllables(word: str) -> int:
    """Count syllables in a word (simplified)."""
    word = word.lower()
    vowels = 'aeiouy'
    count = 0
    prev_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    return max(1, count)


class CurriculumScheduler:
    """Orders training data from easy to hard.

    Strategies:
      - 'vanilla': strict easy→hard ordering
      - 'pacing': gradually increase difficulty
      - 'interleaved': mix easy and hard with increasing hard ratio
      - 'warmup': easy warmup then random sampling
    """

    def __init__(self, strategy: str = "pacing",
                 warmup_ratio: float = 0.1):
        self.strategy = strategy
        self.warmup_ratio = warmup_ratio
        self._scores: list[tuple[int, DifficultyScore]] = []
        self._order: list[int] = []

    def score_samples(self, samples: list[str]) -> list[DifficultyScore]:
        """Compute difficulty scores for all samples."""
        scores = []
        for text in samples:
            scores.append(compute_difficulty(text))
        return scores

    def build_curriculum(self, samples: list[str]) -> list[int]:
        """Build curriculum ordering for samples.

        Returns list of indices in curriculum order.
        """
        scores = self.score_samples(samples)
        self._scores = list(enumerate(scores))

        # Sort by difficulty (easy → hard)
        sorted_indices = sorted(range(len(scores)),
                                key=lambda i: scores[i].combined)

        if self.strategy == "vanilla":
            self._order = sorted_indices
        elif self.strategy == "pacing":
            # Gradually increase difficulty: start with easiest 20%,
            # then 40%, then 60%, etc.
            self._order = self._pacing_order(sorted_indices, len(samples))
        elif self.strategy == "interleaved":
            # Mix easy and hard, increasing hard ratio over time
            self._order = self._interleaved_order(sorted_indices, len(samples))
        elif self.strategy == "warmup":
            # Easy warmup (warmup_ratio of data), then random
            n_warmup = int(len(samples) * self.warmup_ratio)
            warmup = sorted_indices[:n_warmup]
            rest = sorted_indices[n_warmup:]
            random.shuffle(rest)
            self._order = warmup + rest
        else:
            self._order = list(range(len(samples)))

        return self._order

    def _pacing_order(self, sorted_indices: list[int], n: int) -> list[int]:
        """Pacing: gradually expose harder samples."""
        order = []
        # Phase 1: easiest 20%
        phase1_end = n // 5
        order.extend(sorted_indices[:phase1_end])
        # Phase 2: easiest 40%
        phase2_end = 2 * n // 5
        order.extend(sorted_indices[phase1_end:phase2_end])
        # Phase 3: easiest 60%
        phase3_end = 3 * n // 5
        order.extend(sorted_indices[phase2_end:phase3_end])
        # Phase 4: easiest 80%
        phase4_end = 4 * n // 5
        order.extend(sorted_indices[phase3_end:phase4_end])
        # Phase 5: all
        order.extend(sorted_indices[phase4_end:])
        return order

    def _interleaved_order(self, sorted_indices: list[int], n: int) -> list[int]:
        """Interleaved: mix easy and hard, increasing hard ratio."""
        easy = sorted_indices[:n // 2]
        hard = sorted_indices[n // 2:]
        random.shuffle(easy)
        random.shuffle(hard)

        order = []
        easy_idx = 0
        hard_idx = 0
        # Start with 80% easy, 20% hard, gradually shift to 20% easy, 80% hard
        for i in range(n):
            hard_ratio = i / n  # increases from 0 to 1
            if random.random() < hard_ratio and hard_idx < len(hard):
                order.append(hard[hard_idx])
                hard_idx += 1
            elif easy_idx < len(easy):
                order.append(easy[easy_idx])
                easy_idx += 1
            else:
                order.append(hard[hard_idx])
                hard_idx += 1
        return order

    def stats(self) -> dict:
        if not self._scores:
            return {}
        scores = [s.combined for _, s in self._scores]
        return {
            "strategy": self.strategy,
            "n_samples": len(self._scores),
            "difficulty_range": (min(scores), max(scores)),
            "difficulty_mean": sum(scores) / len(scores),
        }


# ─── Training-Time Data Augmentation ───

class DataAugmentor:
    """Training-time augmentation for data-constrained pretraining.

    Three orthogonal categories (from arXiv 2606.16246):
      1. Token-level noise: masking, random replacement
      2. Sequence permutations: right-to-left prediction, Fill-in-the-Middle
      3. Target offset prediction: predict x_{t+i} for i > 1

    These regularize against overfitting in multi-epoch training.
    """

    def __init__(self, mask_prob: float = 0.15,
                 replace_prob: float = 0.05,
                 fim_prob: float = 0.1,
                 offset_probs: dict[int, float] | None = None):
        self.mask_prob = mask_prob
        self.replace_prob = replace_prob
        self.fim_prob = fim_prob
        self.offset_probs = offset_probs or {2: 0.05, 3: 0.02}

    def augment(self, input_ids: torch.Tensor,
                vocab_size: int = 65536) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply random augmentation to a sequence.

        Returns (augmented_input, target) where target may be offset
        from the standard next-token target.
        """
        T = input_ids.shape[1]
        augmented = input_ids.clone()

        # 1. Token-level noise
        if random.random() < self.mask_prob:
            # Mask random tokens
            mask = torch.rand(T) < self.mask_prob
            # Use a special mask token (or just a random token)
            augmented[0, mask] = 0  # <pad> as mask

        if random.random() < self.replace_prob:
            # Random replacement
            replace = torch.rand(T) < self.replace_prob
            augmented[0, replace] = torch.randint(0, vocab_size, (replace.sum(),))

        # 2. Fill-in-the-Middle (FIM)
        if random.random() < self.fim_prob and T > 10:
            augmented = self._apply_fim(augmented)

        # 3. Target offset prediction
        offset = 1  # default: next token
        for k, prob in self.offset_probs.items():
            if random.random() < prob:
                offset = k
                break

        # Create target with offset (guard against T <= offset)
        if offset > 1 and offset + 1 < T:
            target = torch.cat([input_ids[0, offset:],
                                torch.zeros(offset, dtype=torch.long,
                                           device=input_ids.device)])
            target = target.unsqueeze(0)
        elif T > 1:
            target = torch.cat([input_ids[0, 1:],
                                torch.zeros(1, dtype=torch.long,
                                           device=input_ids.device)])
            target = target.unsqueeze(0)
        else:
            # T <= 1: return unchanged
            target = input_ids

        return augmented, target

    def _apply_fim(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Apply Fill-in-the-Middle augmentation.

        Splits the sequence into prefix, middle, suffix and rearranges:
        <prefix> <middle> <suffix> → <pre> <prefix> <suf> <suffix> <mid> <middle>
        """
        T = input_ids.shape[1]
        if T < 8:
            # Too short for FIM — return unchanged
            return input_ids
        # Random split points (guaranteed valid ranges)
        split1 = random.randint(max(1, T // 4), T // 2)
        split2 = random.randint(split1 + 1, max(split1 + 1, 3 * T // 4))

        prefix = input_ids[0, :split1]
        middle = input_ids[0, split1:split2]
        suffix = input_ids[0, split2:]

        # Rearrange (simplified: just swap middle and suffix)
        rearranged = torch.cat([prefix, suffix, middle]).unsqueeze(0)
        return rearranged


# ─── SYNPRO Synthetic Data Generation ───

class SYNPROGenerator:
    """SYNPRO: synthetic data generation for data-bound scaling.

    Two operations:
      1. Rephrasing: present the same content in different words
      2. Reformat: present the same content in a different format
         (e.g., prose → bullet points, Q&A, tutorial)

    Both are optimized via RL with quality, faithfulness, and data influence
    rewards. Continuously updated as pretraining plateaus.

    For our setup: use the model itself to generate rephrasings of training data.
    """

    def __init__(self, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer

    def rephrase(self, text: str, n_variants: int = 3) -> list[str]:
        """Generate rephrasings of the input text."""
        if self.model is None or self.tokenizer is None:
            # Fallback: simple word-level rephrasing
            return [self._simple_rephrase(text) for _ in range(n_variants)]

        variants = []
        for _ in range(n_variants):
            prompt = f"Rephrase the following text, preserving meaning:\n{text}\n\nRephrased:"
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt")
                device = next(self.model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.inference_mode():
                    output = self.model.generate(**inputs, max_new_tokens=256,
                                                  temperature=0.8, do_sample=True)
                variant = self.tokenizer.decode(output[0], skip_special_tokens=True)
                variants.append(variant)
            except Exception:
                variants.append(self._simple_rephrase(text))
        return variants

    def reformat(self, text: str, format: str = "bullets") -> str:
        """Reformat text into a different structure.

        Formats: 'bullets', 'qa', 'tutorial', 'summary'
        """
        if self.model is None or self.tokenizer is None:
            return self._simple_reformat(text, format)

        format_prompts = {
            "bullets": "Convert to bullet points:\n",
            "qa": "Convert to Q&A format:\n",
            "tutorial": "Convert to a step-by-step tutorial:\n",
            "summary": "Summarize:\n",
        }

        prompt = f"{format_prompts.get(format, 'Reformat: ')}{text}\n\nOutput:"
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.inference_mode():
                output = self.model.generate(**inputs, max_new_tokens=256,
                                              temperature=0.3, do_sample=True)
            return self.tokenizer.decode(output[0], skip_special_tokens=True)
        except Exception:
            return self._simple_reformat(text, format)

    def _simple_rephrase(self, text: str) -> str:
        """Simple word-level rephrasing (fallback)."""
        words = text.split()
        # Swap some synonyms (simplified)
        synonyms = {
            "the": "the", "a": "one", "is": "was", "are": "were",
            "will": "shall", "can": "could", "make": "create",
            "use": "utilize", "show": "display", "get": "obtain",
        }
        rephrased = [synonyms.get(w.lower(), w) for w in words]
        return " ".join(rephrased)

    def _simple_reformat(self, text: str, format: str) -> str:
        """Simple reformatting (fallback)."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if format == "bullets":
            return "\n".join(f"• {s}" for s in sentences)
        elif format == "qa":
            return "\n".join(f"Q: What about {s[:30]}?\nA: {s}" for s in sentences[:3])
        elif format == "summary":
            return " ".join(sentences[:2])
        return text

    def generate_batch(self, texts: list[str], n_per_text: int = 2) -> list[str]:
        """Generate synthetic variants for a batch of texts."""
        synthetic = []
        for text in texts:
            # Rephrasing
            variants = self.rephrase(text, n_variants=n_per_text // 2)
            synthetic.extend(variants)
            # Reformatting
            reformatted = self.reformat(text, format="bullets")
            synthetic.append(reformatted)
        return synthetic
