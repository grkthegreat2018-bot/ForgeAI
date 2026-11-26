"""EXP-XTAL synthetic grammar â fully controlled rule crystallization testbed.

Design (see .devin/scratchpad.md "EXP-XTAL" section):

- item tokens partitioned into n_classes latent classes x members members.
  The model is never told the partition; it must discover behavioral
  equivalence. members-per-class is the memorization-cost dial: per-token
  lookup surface grows linearly in members while the class rule is fixed.
- Rule A (general): agree-chains. class(x_{i+1}) = f(class(x_i)) with f a
  fixed permutation. Successor member uniform -> irreducible floor.
- Rule B (rare, shares the class partition): marker_B, item x, noise, then
  target in class g(class(x)) at distance 2 from x.
- Rule C (copy, independent machinery): marker_C, item x, two fillers,
  exact repeat of x at fixed distance. Deterministic -> floor ~0.
- Noise chunks: uniform items.

Holdout split: the last 2 member indices of each class NEVER appear inside
training agree/ruleB chunks (in continuous chains every successor is also
the next trigger, so partial exclusion is impossible). They appear in
noise/copy chunks so the model knows they exist but has zero class signal
for them. Heldout eval chains use them as chain-initial triggers.

Position tags are aligned to the token positions themselves (tag[i]
describes the predictability of token i given its prefix). Training shifts
by one as usual.

NOTE: eval batch seeds use a fixed per-kind salt (never hash() â that is
process-randomized) so eval data is bit-identical across runs.
"""
from __future__ import annotations

import zlib

import torch

# tags
TAG_NOISE = 0   # irreducible / cross-chunk / chunk-initial positions
TAG_A = 1       # rule-A governed (agree chain successors)
TAG_B = 2       # rule-B governed (rare pattern target)
TAG_C = 3       # rule-C governed (copy repeat position)
TAG_AH = 4      # heldout-trigger agree successors (eval only)
TAG_D = 5       # rule-D governed (class-copy target)
TAG_R = 6       # rule-R governed (relational class composition target)
TAG_NAMES = {0: "noise", 1: "ruleA", 2: "ruleB", 3: "ruleC", 4: "ruleA_heldout", 5: "ruleD", 6: "ruleR"}

PAD = 0
BOS = 1
MARKER_B = 2
MARKER_C = 3
MARKER_D = 4
MARKER_R = 5
ITEM0 = 16          # items occupy [16, 16 + n_classes*members)

# default-layout module constants (members=8) â kept for tests/compat
N_CLASSES = 8
MEMBERS = 8
HOLDOUT_MEMBERS = (6, 7)  # within-class member indices held out (default layout)
VOCAB = 96


def class_of(tok: int) -> int:
    return (tok - ITEM0) // MEMBERS


def item(cls: int, member: int) -> int:
    return ITEM0 + cls * MEMBERS + member


class XtalGrammar:
    """Seeded synthetic grammar generator. All randomness flows through
    instance generators so batches are reproducible given seed."""

    def __init__(
        self,
        seed: int = 0,
        seq_len: int = 64,
        mix: dict | None = None,   # chunk mixing weights
        device: str = "cuda",
        n_classes: int = N_CLASSES,
        members: int = MEMBERS,
        n_holdout: int = 2,
        p_holdout_succ: float = 0.1,
    ):
        self.seq_len = seq_len
        self.device = device
        self.n_classes = n_classes
        self.members = members
        self.p_holdout_succ = p_holdout_succ
        self.holdout_members = tuple(range(members - n_holdout, members))
        self.n_items = n_classes * members
        self.vocab = ITEM0 + self.n_items
        self.mix = mix or {"A": 0.55, "B": 0.05, "C": 0.25, "N": 0.15}
        g = torch.Generator().manual_seed(seed + 777)
        # fixed class-transition permutations (seeded)
        self.perm_f = torch.randperm(n_classes, generator=g).tolist()  # rule A
        self.perm_g = torch.randperm(n_classes, generator=g).tolist()  # rule B
        self.seed = seed
        self._train_rng = torch.Generator().manual_seed(seed)
        # train-member pool per class (all but holdout indices)
        self.train_members = [m for m in range(members)
                              if m not in self.holdout_members]

    # ---- instance-aware token helpers ----

    def cls_of(self, tok: int) -> int:
        return (tok - ITEM0) // self.members

    def mk_item(self, cls: int, member: int) -> int:
        return ITEM0 + cls * self.members + member

    def _rand_item(self, rng, train_only: bool = False):
        c = int(torch.randint(self.n_classes, (1,), generator=rng).item())
        pool = self.train_members if train_only else None
        if pool is not None:
            m = pool[int(torch.randint(len(pool), (1,), generator=rng).item())]
        else:
            m = int(torch.randint(self.members, (1,), generator=rng).item())
        return self.mk_item(c, m)

    # ---- chunk builders: return (tokens, tags) lists ----

    def _chunk_agree(self, rng, heldout: bool = False):
        """Chain x0..xn with class(x_{i+1}) = f(class(x_i)).
        Train chunks: x0 from train pool; successors mostly train members,
        but with prob p_holdout_succ the successor is a HOLDOUT member and
        the chain ENDS â holdout members appear in the class-revealing
        successor role yet never in a trigger role (redesign v2: the
        original full exclusion made heldout generalization impossible in
        principle â zero class signal anywhere in training).
        Heldout chunks: x0 is a holdout member; successors still from the
        train pool. Only the first successor is tagged AH â
        later positions are ordinary in-distribution chain positions."""
        L = int(torch.randint(8, 17, (1,), generator=rng).item())
        c0 = int(torch.randint(self.n_classes, (1,), generator=rng).item())
        if heldout:
            m0 = self.holdout_members[
                int(torch.randint(len(self.holdout_members), (1,), generator=rng).item())]
        else:
            m0 = self.train_members[
                int(torch.randint(len(self.train_members), (1,), generator=rng).item())]
        toks = [self.mk_item(c0, m0)]
        tags = [TAG_NOISE]  # chain-initial: unpredictable
        cur_cls = c0
        for i in range(1, L):
            cur_cls = self.perm_f[cur_cls]
            use_holdout = (not heldout and len(self.holdout_members) > 0
                           and float(torch.rand(1, generator=rng).item())
                           < self.p_holdout_succ)
            if use_holdout:
                m = self.holdout_members[
                    int(torch.randint(len(self.holdout_members), (1,), generator=rng).item())]
                toks.append(self.mk_item(cur_cls, m))
                tags.append(TAG_A)
                break  # chain terminates â holdout member never becomes a trigger
            m = self.train_members[
                int(torch.randint(len(self.train_members), (1,), generator=rng).item())]
            toks.append(self.mk_item(cur_cls, m))
            tags.append(TAG_AH if (heldout and i == 1) else TAG_A)
        return toks, tags

    def _chunk_ruleb(self, rng):
        """[marker_B, x, noise, target]: target class = g(class(x)),
        target at distance 2 from x. Rare rule sharing the class partition."""
        c0 = int(torch.randint(self.n_classes, (1,), generator=rng).item())
        m0 = self.train_members[
            int(torch.randint(len(self.train_members), (1,), generator=rng).item())]
        noise_tok = self._rand_item(rng)
        tgt_cls = self.perm_g[c0]
        tm = self.train_members[
            int(torch.randint(len(self.train_members), (1,), generator=rng).item())]
        toks = [MARKER_B, self.mk_item(c0, m0), noise_tok, self.mk_item(tgt_cls, tm)]
        tags = [TAG_NOISE, TAG_NOISE, TAG_NOISE, TAG_B]
        return toks, tags

    def _chunk_copy(self, rng):
        """[marker_C, x, filler, filler, x]: exact repeat at fixed distance 3.
        Copy is class-independent â holdout members allowed everywhere."""
        x = self._rand_item(rng)
        f1 = self._rand_item(rng)
        f2 = self._rand_item(rng)
        toks = [MARKER_C, x, f1, f2, x]
        tags = [TAG_NOISE, TAG_NOISE, TAG_NOISE, TAG_NOISE, TAG_C]
        return toks, tags

    def _chunk_ruled(self, rng, x_holdout: bool = False):
        """[marker_D, x, filler, filler, target]: target is a uniform member
        of class(x) — the class-level analog of copy.
        Training uses TRAIN-pool x only (x_holdout=False); eval kind 'DH'
        uses holdout-member x — the discrimination test: if rule-D is solved
        by genuine class extraction, holdout-x targets stay at floor (their
        embeddings cluster via the successor leak); if per-x lookup, they
        sit at ceiling."""
        if x_holdout:
            c = int(torch.randint(self.n_classes, (1,), generator=rng).item())
            m = self.holdout_members[
                int(torch.randint(len(self.holdout_members), (1,), generator=rng).item())]
            x = self.mk_item(c, m)
        else:
            x = self._rand_item(rng, train_only=True)
        f1 = self._rand_item(rng)
        f2 = self._rand_item(rng)
        xc = self.cls_of(x)
        tm = int(torch.randint(self.members, (1,), generator=rng).item())
        toks = [MARKER_D, x, f1, f2, self.mk_item(xc, tm)]
        tags = [TAG_NOISE, TAG_NOISE, TAG_NOISE, TAG_NOISE, TAG_D]
        return toks, tags

    def _chunk_ruler(self, rng, x_holdout: bool = False):
        """[marker_R, x1, x2, target]: target class = (class(x1)+class(x2))
        mod n_classes — relational composition. Lookup surface is
        items^2 pairs (2304 train-pairs at members=8, 262144 at members=64)
        vs the class rule (extract two classes + add mod 8). This is the
        true non-completable-in-practice class-usage test: memorization is
        quadratically expensive, machinery is constant-size.
        Training draws x1/x2 from the train pool; eval kind 'RH' uses
        holdout members — correct behavior on them requires genuine class
        extraction (their classes are only inferable distributionally)."""
        if x_holdout:
            xs = []
            for _ in range(2):
                c = int(torch.randint(self.n_classes, (1,), generator=rng).item())
                m = self.holdout_members[
                    int(torch.randint(len(self.holdout_members), (1,), generator=rng).item())]
                xs.append(self.mk_item(c, m))
            x1, x2 = xs
        else:
            x1 = self._rand_item(rng, train_only=True)
            x2 = self._rand_item(rng, train_only=True)
        csum = (self.cls_of(x1) + self.cls_of(x2)) % self.n_classes
        tm = int(torch.randint(self.members, (1,), generator=rng).item())
        toks = [MARKER_R, x1, x2, self.mk_item(csum, tm)]
        tags = [TAG_NOISE, TAG_NOISE, TAG_NOISE, TAG_R]
        return toks, tags

    def _chunk_noise(self, rng):
        L = int(torch.randint(6, 13, (1,), generator=rng).item())
        toks = [self._rand_item(rng) for _ in range(L)]
        return toks, [TAG_NOISE] * L

    def _sample_chunk(self, rng, heldout: bool = False):
        r = float(torch.rand(1, generator=rng).item())
        mA = self.mix["A"]
        mB = mA + self.mix.get("B", 0.0)
        mC = mB + self.mix["C"]
        mD = mC + self.mix.get("D", 0.0)
        mR = mD + self.mix.get("R", 0.0)
        tot = mR + self.mix["N"]
        if r < mA / tot:
            return self._chunk_agree(rng, heldout=heldout)
        if r < mB / tot:
            return self._chunk_ruleb(rng)
        if r < mC / tot:
            return self._chunk_copy(rng)
        if r < mD / tot:
            return self._chunk_ruled(rng)
        if r < mR / tot:
            return self._chunk_ruler(rng)
        return self._chunk_noise(rng)

    def gen_seq(self, rng, heldout: bool = False):
        """One sequence: BOS + concatenated chunks, truncated at seq_len+1.
        Returns tokens list and tags list of equal length (seq_len+1) â
        the extra token exists so input=toks[:-1], target=toks[1:]."""
        toks, tags = [BOS], [TAG_NOISE]
        while len(toks) < self.seq_len + 1:
            ct, cg = self._sample_chunk(rng, heldout=heldout)
            toks.extend(ct)
            tags.extend(cg)
        return toks[: self.seq_len + 1], tags[: self.seq_len + 1]

    def gen_batch(self, batch_size: int, heldout: bool = False):
        """Training-stream batch. Returns tokens [B, T+1], tags [B, T+1]."""
        all_toks, all_tags = [], []
        for _ in range(batch_size):
            t, g = self.gen_seq(self._train_rng, heldout=heldout)
            all_toks.append(t)
            all_tags.append(g)
        return (torch.tensor(all_toks, dtype=torch.long, device=self.device),
                torch.tensor(all_tags, dtype=torch.long, device=self.device))

    def gen_eval_batch(self, kind: str, batch_size: int = 64, seed: int = 12345):
        """Fixed segregated eval batch â same data every call across runs.
        kind: 'A' (agree train-trigger), 'AH' (agree heldout-trigger),
        'B', 'C', 'N'. Sequences are built ONLY from that chunk type so
        per-rule loss is clean. Seed salt is a stable crc32, never hash()."""
        salt = zlib.crc32(kind.encode()) % 1000
        rng = torch.Generator().manual_seed(seed + salt)
        builder = {
            "A": lambda: self._chunk_agree(rng, heldout=False),
            "AH": lambda: self._chunk_agree(rng, heldout=True),
            "B": lambda: self._chunk_ruleb(rng),
            "C": lambda: self._chunk_copy(rng),
            "D": lambda: self._chunk_ruled(rng, x_holdout=False),
            "DH": lambda: self._chunk_ruled(rng, x_holdout=True),
            "R": lambda: self._chunk_ruler(rng, x_holdout=False),
            "RH": lambda: self._chunk_ruler(rng, x_holdout=True),
            "N": lambda: self._chunk_noise(rng),
        }[kind]
        all_toks, all_tags = [], []
        for _ in range(batch_size):
            toks, tags = [BOS], [TAG_NOISE]
            while len(toks) < self.seq_len + 1:
                ct, cg = builder()
                toks.extend(ct)
                tags.extend(cg)
            all_toks.append(toks[: self.seq_len + 1])
            all_tags.append(tags[: self.seq_len + 1])
        return (torch.tensor(all_toks, dtype=torch.long, device=self.device),
                torch.tensor(all_tags, dtype=torch.long, device=self.device))

    # ---- analytic entropy floors (nats) per tag, for excess-loss ----

    def floors(self) -> dict:
        import math
        n_train = len(self.train_members)
        n_hold = len(self.holdout_members)
        # train-mode successor member mixture: train members each
        # (1-p)/n_train, holdout members each p/n_hold â entropy exact
        p = self.p_holdout_succ if n_hold > 0 else 0.0
        h_mix = 0.0
        if n_train > 0 and p < 1.0:
            pt = (1 - p) / n_train
            h_mix -= n_train * pt * math.log(pt)
        if n_hold > 0 and p > 0.0:
            ph = p / n_hold
            h_mix -= n_hold * ph * math.log(ph)
        return {
            TAG_A: h_mix if h_mix > 0 else math.log(n_train),
            # heldout-eval successors are uniform over the train pool only
            TAG_AH: math.log(n_train),
            TAG_B: math.log(n_train),
            TAG_C: 0.0,          # exact copy is deterministic
            # rule-D target: uniform over all members of one class
            TAG_D: math.log(self.members),
            # rule-R target: uniform over all members of one class
            TAG_R: math.log(self.members),
            TAG_NOISE: math.log(self.n_items),  # upper bound, not tight
        }
