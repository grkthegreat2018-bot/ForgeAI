"""Generate non-tool SFT examples so the model learns when NOT to use tools.

Mix these with Glaive tool-use data to prevent overfitting to tool-call format.
"""
import json
import os
import random

random.seed(42)

# Diverse non-tool Q&A pairs — general knowledge, coding, reasoning
# These teach the model: "not every question needs a tool call"
NON_TOOL_EXAMPLES = [
    {"messages": [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2 + 2 = 4."},
    ]},
    {"messages": [
        {"role": "user", "content": "Explain how a transformer attention mechanism works in 2 sentences."},
        {"role": "assistant", "content": "Transformer attention computes weighted sums of value vectors, where weights are derived from query-key dot products scaled by the square root of dimension. This allows each token to focus on relevant context positions, enabling long-range dependency modeling."},
    ]},
    {"messages": [
        {"role": "user", "content": "Write a Python function to reverse a list."},
        {"role": "assistant", "content": "def reverse_list(lst):\n    return lst[::-1]"},
    ]},
    {"messages": [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
    ]},
    {"messages": [
        {"role": "user", "content": "What's the difference between a list and a tuple in Python?"},
        {"role": "assistant", "content": "Lists are mutable (can be modified after creation) while tuples are immutable (cannot be changed). Tuples are also slightly faster and use less memory. Use tuples for fixed data and lists for data that needs to change."},
    ]},
    {"messages": [
        {"role": "user", "content": "How does photosynthesis work?"},
        {"role": "assistant", "content": "Photosynthesis is the process by which plants convert light energy into chemical energy. Chlorophyll in the leaves absorbs sunlight, which is used to convert carbon dioxide and water into glucose and oxygen. The glucose feeds the plant and oxygen is released as a byproduct."},
    ]},
    {"messages": [
        {"role": "user", "content": "Write a haiku about autumn."},
        {"role": "assistant", "content": "Golden leaves falling\nCrisp air whispers winter near\nBranches bare and still"},
    ]},
    {"messages": [
        {"role": "user", "content": "What is recursion?"},
        {"role": "assistant", "content": "Recursion is when a function calls itself to solve a smaller instance of the same problem. Each recursive call reduces the problem size until it reaches a base case that can be solved directly. For example, factorial(n) = n * factorial(n-1), with factorial(0) = 1 as the base case."},
    ]},
    {"messages": [
        {"role": "user", "content": "Explain quantum entanglement simply."},
        {"role": "assistant", "content": "Quantum entanglement is when two particles become linked so that measuring one instantly affects the other, regardless of distance. Einstein called it 'spooky action at a distance'. The particles share a quantum state, so knowing one tells you about the other without any signal traveling between them."},
    ]},
    {"messages": [
        {"role": "user", "content": "What is the Fibonacci sequence?"},
        {"role": "assistant", "content": "The Fibonacci sequence is a series where each number is the sum of the two preceding ones: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ... It appears in nature (spiral shells, flower petals) and is used in algorithms, finance, and art."},
    ]},
    {"messages": [
        {"role": "user", "content": "How do I sort a dictionary by value in Python?"},
        {"role": "assistant", "content": "sorted_dict = dict(sorted(my_dict.items(), key=lambda x: x[1]))"},
    ]},
    {"messages": [
        {"role": "user", "content": "What causes the seasons?"},
        {"role": "assistant", "content": "The seasons are caused by Earth's axial tilt (23.5 degrees) as it orbits the sun. When a hemisphere tilts toward the sun, it gets more direct sunlight and experiences summer. When it tilts away, it gets less sunlight and experiences winter."},
    ]},
    {"messages": [
        {"role": "user", "content": "What is machine learning?"},
        {"role": "assistant", "content": "Machine learning is a subset of AI where computers learn patterns from data instead of being explicitly programmed. The model is trained on examples, finds patterns, and uses them to make predictions on new data. Common types include supervised learning (labeled data), unsupervised learning (unlabeled data), and reinforcement learning (reward-based)."},
    ]},
    {"messages": [
        {"role": "user", "content": "Write a function to check if a string is a palindrome."},
        {"role": "assistant", "content": "def is_palindrome(s):\n    s = s.lower().replace(' ', '')\n    return s == s[::-1]"},
    ]},
    {"messages": [
        {"role": "user", "content": "What is the speed of light?"},
        {"role": "assistant", "content": "The speed of light in a vacuum is approximately 299,792,458 meters per second (about 300,000 km/s). This is a fundamental constant of nature, denoted as 'c' in physics."},
    ]},
    {"messages": [
        {"role": "user", "content": "Explain the difference between TCP and UDP."},
        {"role": "assistant", "content": "TCP (Transmission Control Protocol) is connection-oriented, reliable, and ordered — it guarantees delivery and reassembles data in order. UDP (User Datagram Protocol) is connectionless, unreliable, and unordered — it sends packets without guaranteeing delivery or order, but is faster and has lower overhead. TCP is used for web browsing, email; UDP for streaming, gaming, DNS."},
    ]},
    {"messages": [
        {"role": "user", "content": "What is a neural network?"},
        {"role": "assistant", "content": "A neural network is a computational model inspired by the brain, consisting of layers of interconnected nodes (neurons). Each connection has a weight that's learned during training. Data passes through layers, each applying a mathematical transformation, to produce an output. Deep neural networks have many layers and can learn complex patterns."},
    ]},
    {"messages": [
        {"role": "user", "content": "How do I read a file in Python?"},
        {"role": "assistant", "content": "with open('file.txt', 'r') as f:\n    content = f.read()"},
    ]},
    {"messages": [
        {"role": "user", "content": "What is the meaning of life?"},
        {"role": "assistant", "content": "That's a philosophical question with no single answer. Different perspectives: biology says it's reproduction and survival, philosophy says it's finding purpose and meaning, religion offers spiritual answers. Ultimately, many people find meaning through relationships, creativity, learning, and helping others."},
    ]},
    {"messages": [
        {"role": "user", "content": "What is git and why is it useful?"},
        {"role": "assistant", "content": "Git is a version control system that tracks changes in code. It lets developers save snapshots (commits), branch off for features, merge work together, and revert to previous states. It's useful for collaboration, backup, and maintaining project history."},
    ]},
]

# Duplicate with slight variations to get enough non-tool examples
all_examples = []
for _ in range(50):
    for ex in NON_TOOL_EXAMPLES:
        all_examples.append(dict(ex))

random.shuffle(all_examples)

output_path = "data/sft/nontool_general.jsonl"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    for ex in all_examples:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"Written {len(all_examples)} non-tool examples to {output_path}")
