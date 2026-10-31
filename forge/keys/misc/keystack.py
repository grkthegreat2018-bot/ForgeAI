"""KeyStack — compose multiple keys into a full model key.

A KeyStack stitches component keys together using the model's forward pass
wiring. Each key handles one component; the stack handles the data flow
between them.

Building a KeyStack:
  1. List the component keys in forward order
  2. Provide stitching: how each key's output feeds the next
  3. Use dummy data to verify the stitching works
  4. The stack can then process real data/weights end-to-end
"""

import torch

from forge.keys.misc.base import Key, KeyClass, KeyResult


class KeyStack:
    """A stack of keys that processes a full model.

    Keys are applied in order. Each key receives data from the previous key's
    output (or the initial input) and produces weights/data for its component.
    """

    def __init__(self, name: str = "unnamed"):
        self.name = name
        self.keys: list[Key] = []
        self.stitching: dict[str, str] = {}  # output_name -> input_name mapping

    def add(self, key: Key) -> 'KeyStack':
        """Add a key to the stack."""
        self.keys.append(key)
        return self

    def describe(self) -> str:
        """Human-readable description of the stack."""
        lines = [f"KeyStack: {self.name}", f"  Components ({len(self.keys)}):"]
        for i, key in enumerate(self.keys):
            lines.append(f"    {i+1}. {key.name} [{key.key_class().value}]")
        bi_count = sum(1 for k in self.keys if k.key_class() in (KeyClass.BI, KeyClass.FULL))
        partial_count = sum(1 for k in self.keys if k.key_class() == KeyClass.PARTIAL)
        trivial_count = sum(1 for k in self.keys if k.key_class() == KeyClass.TRIVIAL)
        lines.append(f"  Summary: {bi_count} Bi, {partial_count} Partial, {trivial_count} Trivial")
        all_bi = partial_count == 0
        lines.append(f"  Full Bi KeyStack: {'YES' if all_bi else 'NO (has Partial keys)'}")
        return "\n".join(lines)

    def forward(self, data: dict[str, torch.Tensor]) -> KeyResult:
        """data -> weights for ALL components in the stack.

        Calls each key's forward() with the relevant subset of data.
        Merges all weight outputs into one dict.
        """
        all_weights = {}
        all_metadata = {}
        errors = []

        for key in self.keys:
            # Filter data to what this key needs (pass everything, let key pick)
            result = key.forward(data)
            if result.success:
                all_weights.update(result.weights)
                all_metadata[key.name] = result.metadata
            else:
                errors.append(f"{key.name}: {result.error}")

        if errors and not all_weights:
            return KeyResult(success=False, error="; ".join(errors))

        return KeyResult(
            success=True,
            weights=all_weights,
            metadata={'stack': self.name, 'components': all_metadata,
                     'errors': errors if errors else None}
        )

    def reverse(self, weights: dict[str, torch.Tensor]) -> KeyResult:
        """weights -> data for ALL components."""
        all_data = {}
        all_metadata = {}
        errors = []

        for key in self.keys:
            # Filter weights to what this key owns
            result = key.reverse(weights)
            if result.success:
                all_data.update(result.data)
                all_metadata[key.name] = result.metadata
            else:
                errors.append(f"{key.name}: {result.error}")

        return KeyResult(
            success=bool(all_data),
            data=all_data,
            metadata={'stack': self.name, 'components': all_metadata,
                     'errors': errors if errors else None}
        )

    def cross_arch(self, weights_a: dict[str, torch.Tensor],
                   stack_b: 'KeyStack') -> KeyResult:
        """weights(A) -> data -> weights(B) using two KeyStacks."""
        # Decode A
        decode = self.reverse(weights_a)
        if not decode.success:
            return KeyResult(success=False, error=f"Reverse stack A failed: {decode.error}")

        # Encode B
        encode = stack_b.forward(decode.data)
        if not encode.success:
            return KeyResult(success=False, error=f"Forward stack B failed: {encode.error}")

        return KeyResult(
            success=True,
            weights=encode.weights,
            metadata={'source_stack': self.name, 'target_stack': stack_b.name,
                     'intermediate_keys': list(decode.data.keys())}
        )

    def __repr__(self):
        return f"KeyStack({self.name}, {len(self.keys)} keys)"
