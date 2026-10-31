"""R31 v3: Generate training data from hermes_fc + discovery tools + made-up tools.

Sources:
1. hermes_fc.jsonl (1742 real FC examples) → convert to pythonic format
2. Discovery tools (18 tools from discovery_tools.py) → synthetic examples
3. Made-up tools (fake schemas) → teaches schema adaptation
"""
import json
import random
from pathlib import Path

HF_DATASETS = Path(r"D:\windsurf\ForgeAI\forge\distillation\hf_datasets")
OUT = Path(r"D:\windsurf\ForgeAI\research\data\finetune\r31_v3_training.jsonl")

TCS = "<|tool_call_start|>"
TCE = "<|tool_call_end|>"
TRS = "<|tool_response_start|>"
TRE = "<|tool_response_end|>"

# ── All 18 discovery tools (from discovery_tools.py) ───────────────────────
DISCOVERY_TOOLS = [
    {"name": "think", "description": "Record a train-of-thought idea or observation.",
     "parameters": {"content": "string", "confidence": "number 0-1 optional"}},
    {"name": "sudo_think", "description": "Meta-reason about your own process.",
     "parameters": {"content": "string"}},
    {"name": "run_script", "description": "Execute Python code in a sandbox (8s timeout, no network).",
     "parameters": {"code": "string"}},
    {"name": "web_search", "description": "Search the internet via DuckDuckGo. Returns result snippets.",
     "parameters": {"query": "string", "n": "int optional, default 5"}},
    {"name": "wikipedia_search", "description": "Search Wikipedia for encyclopedic knowledge.",
     "parameters": {"query": "string", "n": "int optional, default 3"}},
    {"name": "arxiv_search", "description": "Search arXiv for academic papers.",
     "parameters": {"query": "string", "n": "int optional, default 3"}},
    {"name": "fetch_url", "description": "Fetch a web page and extract its text content.",
     "parameters": {"url": "string", "max_chars": "int optional, default 2000"}},
    {"name": "calculate", "description": "Evaluate a math expression or short Python calculation.",
     "parameters": {"code": "string"}},
    {"name": "save_research", "description": "Persist a web research finding to your database.",
     "parameters": {"query": "string", "url": "string", "title": "string", "summary": "string", "snippet": "string"}},
    {"name": "propose_theory", "description": "Log a hypothesis to track. Status starts open.",
     "parameters": {"statement": "string", "notes": "string optional"}},
    {"name": "update_theory", "description": "Update a theory's status/evidence.",
     "parameters": {"theory_id": "int", "status": "string optional", "evidence_for": "int optional", "evidence_against": "int optional", "notes": "string optional"}},
    {"name": "record_discovery", "description": "Record a confirmed finding you've verified.",
     "parameters": {"summary": "string", "theory_id": "int optional", "confidence": "number 0-1 optional"}},
    {"name": "query_db", "description": "Read-only SELECT against your own memory.",
     "parameters": {"sql": "string", "params": "list optional"}},
    {"name": "migrate_schema", "description": "Add/refactor database tables (CREATE/ALTER/INDEX/VIEW only).",
     "parameters": {"sql": "string", "reason": "string"}},
    {"name": "summarize_context", "description": "Summarize what you've learned so far.",
     "parameters": {"summary": "string", "confidence": "number 0-1 optional"}},
    {"name": "finish_session", "description": "End this discovery session with a summary.",
     "parameters": {"summary": "string"}},
    {"name": "set_goal", "description": "Set your own goal for this session.",
     "parameters": {"goal": "string"}},
    {"name": "ask_clarification", "description": "Ask a clarifying question about the task.",
     "parameters": {"question": "string"}},
]

def tool_list_str(tools):
    parts = [json.dumps(t, ensure_ascii=False) for t in tools]
    return "List of tools: [" + ", ".join(parts) + "]"

def system_prompt(tools=None):
    base = ("You are an autonomous discovery agent with no fixed goal. Explore what interests "
            "you, form theories, test them with code, search the web, and record findings in "
            "your database. You decide what to investigate.\n\n"
            "Your database tables: thoughts, scripts, research, theories, discoveries\n\n"
            "Each turn: write brief reasoning, then call ONE tool. To call a tool, output "
            "the tool call tokens like this example:\n"
            "<|tool_call_start|>[think(content='Primes greater than 5 only end in 1,3,7,9')]<|tool_call_end|>\n\n"
            "Call finish_session when done exploring.")
    if tools:
        base += "\n\n" + tool_list_str(tools)
    return base

def tc(name, **args):
    if not args:
        return f"{TCS}[{name}()]{TCE}"
    parts = []
    for k, v in args.items():
        if isinstance(v, str):
            v = v.replace("\\", "\\\\").replace("'", "\\'")
            parts.append(f"{k}='{v}'")
        elif isinstance(v, bool) or isinstance(v, (int, float)):
            parts.append(f"{k}={v}")
        else:
            parts.append(f"{k}={json.dumps(v)}")
    return f"{TCS}[{name}({', '.join(parts)})]{TCE}"

def tr(d):
    return f"{TRS}{json.dumps(d)}{TRE}"

random.seed(42)
examples = []

# ── 1. Convert hermes_fc.jsonl to pythonic tool-call format ────────────────
# hermes_fc uses: <tool_call>\n{"name": "func", "arguments": {"key": "val"}}\n</tool_call>
# We convert to: <|tool_call_start|>[func(key='val')]<|tool_call_end|>
import re

_hermes_call = re.compile(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', re.DOTALL)

def convert_hermes_call(json_str):
    """Convert {"name": "func", "arguments": {...}} to pythonic func(key='val')."""
    try:
        obj = json.loads(json_str)
        name = obj.get("name", "")
        args = obj.get("arguments", {})
        if not name:
            return None
        return tc(name, **args) if args else tc(name)
    except (json.JSONDecodeError, TypeError):
        return None

def convert_hermes_solution(solution):
    """Convert hermes_fc solution (multiple tool calls) to pythonic format."""
    calls = []
    for m in _hermes_call.finditer(solution):
        converted = convert_hermes_call(m.group(1))
        if converted:
            calls.append(converted)
    if not calls:
        # Try plain JSON format
        try:
            objs = json.loads(solution)
            if isinstance(objs, list):
                for obj in objs:
                    name = obj.get("name", "")
                    args = obj.get("arguments", {})
                    if name:
                        calls.append(tc(name, **args) if args else tc(name))
        except (json.JSONDecodeError, TypeError):
            pass
    return "\n".join(calls) if calls else None

# Generate made-up tool schemas for hermes_fc examples (since we don't have the original schemas)
# This teaches the model to adapt to whatever tools are in the system prompt
made_up_tools_pool = [
    {"name": "get_camera_live_feed", "description": "Get live feed from a camera",
     "parameters": {"camera_id": "string", "stream_quality": "string"}},
    {"name": "record_camera_feed", "description": "Record camera feed",
     "parameters": {"camera_id": "string", "duration": "int"}},
    {"name": "get_recorded_feed", "description": "Get recorded footage",
     "parameters": {"camera_id": "string", "start_time": "string", "end_time": "string"}},
    {"name": "initialize_smart_home_system", "description": "Initialize smart home",
     "parameters": {"device_list": "list"}},
    {"name": "create_device_group", "description": "Create device group",
     "parameters": {"group_name": "string", "devices": "list"}},
    {"name": "send_email", "description": "Send an email",
     "parameters": {"to": "string", "subject": "string", "body": "string"}},
    {"name": "search_products", "description": "Search for products",
     "parameters": {"query": "string", "category": "string", "max_results": "int"}},
    {"name": "book_flight", "description": "Book a flight",
     "parameters": {"origin": "string", "destination": "string", "date": "string", "passengers": "int"}},
    {"name": "get_weather", "description": "Get weather for a location",
     "parameters": {"location": "string", "units": "string"}},
    {"name": "play_music", "description": "Play music",
     "parameters": {"song": "string", "artist": "string", "volume": "int"}},
    {"name": "set_reminder", "description": "Set a reminder",
     "parameters": {"message": "string", "time": "string"}},
    {"name": "calculate_tip", "description": "Calculate restaurant tip",
     "parameters": {"bill_amount": "float", "tip_percentage": "float"}},
    {"name": "convert_currency", "description": "Convert between currencies",
     "parameters": {"amount": "float", "from_currency": "string", "to_currency": "string"}},
    {"name": "find_restaurant", "description": "Find nearby restaurants",
     "parameters": {"cuisine": "string", "location": "string", "radius": "int"}},
    {"name": "schedule_meeting", "description": "Schedule a meeting",
     "parameters": {"title": "string", "date": "string", "duration": "int", "attendees": "list"}},
]

n_hermes = 0
n_hermes_failed = 0
with open(HF_DATASETS / "hermes_fc.jsonl", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        prompt = obj.get("prompt", "")
        solution = obj.get("solution", "")
        if not prompt or not solution:
            continue

        converted = convert_hermes_solution(solution)
        if not converted:
            n_hermes_failed += 1
            continue

        # Pick 3-5 random made-up tools for the system prompt (teaches adaptation)
        n_tools = random.randint(3, 6)
        tools = random.sample(made_up_tools_pool, min(n_tools, len(made_up_tools_pool)))

        msgs = [
            {"role": "system", "content": "You are a helpful assistant. Use the available tools to answer user requests.\n\n" + tool_list_str(tools)},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": converted},
        ]
        examples.append({"messages": msgs})
        n_hermes += 1

print(f"  hermes_fc: {n_hermes} converted, {n_hermes_failed} failed")

# ── 2. Discovery tools synthetic examples (all 18 tools) ───────────────────
disc_sys = system_prompt(DISCOVERY_TOOLS)

# think
thoughts = [
    "Primes greater than 5 only end in 1, 3, 7, or 9.",
    "The Fibonacci sequence converges to the golden ratio 1.618.",
    "A number divisible by 9 has a digit sum divisible by 9.",
    "Even numbers are divisible by 2, odd numbers are not.",
    "A palindrome reads the same forwards and backwards.",
    "The sum of two odd numbers is always even.",
    "Binary search requires a sorted array and has O(log n) complexity.",
    "The factorial of 0 is defined as 1.",
    "The GCD of two numbers can be found using the Euclidean algorithm.",
    "The sum of the first n natural numbers is n*(n+1)/2.",
    "The Pythagorean theorem states a^2 + b^2 = c^2 for right triangles.",
    "The area of a circle is pi times radius squared.",
    "Twin primes are pairs of primes that differ by 2.",
    "The square root of 2 is irrational.",
    "The Collatz conjecture states any positive integer eventually reaches 1.",
    "A composite number has more than two factors.",
    "The harmonic mean of n numbers is n divided by the sum of their reciprocals.",
    "Python strings are immutable.",
    "List comprehensions are faster than for loops in Python.",
    "A hash map provides O(1) average lookup time.",
]
for thought in thoughts:
    for conf in [0.8, 0.9, 0.95]:
        examples.append({"messages": [
            {"role": "system", "content": disc_sys},
            {"role": "user", "content": "Begin exploring."},
            {"role": "assistant", "content": f"Let me record a thought.\n{tc('think', content=thought, confidence=conf)}"},
        ]})

# sudo_think
sudo_thoughts = [
    "I should explore prime numbers next.",
    "My strategy of testing with code is working well.",
    "I need to search for more information on this topic.",
    "Let me try a different approach to this problem.",
    "I should record my findings more carefully.",
]
for thought in sudo_thoughts:
    examples.append({"messages": [
        {"role": "system", "content": disc_sys},
        {"role": "user", "content": "Begin exploring."},
        {"role": "assistant", "content": f"Let me reflect on my approach.\n{tc('sudo_think', content=thought)}"},
    ]})

# run_script
scripts = [
    ("Write a script to print Hello World", "print('Hello World')"),
    ("Write a script to print numbers 1 to 5", "for i in range(1, 6):\n    print(i)"),
    ("Write a script to print even numbers 2 to 10", "for i in range(2, 11, 2):\n    print(i)"),
    ("Write a script to print the first 5 primes", "primes = [2, 3, 5, 7, 11]\nprint(primes)"),
    ("Write a script to compute sum of 1 to 10", "print(sum(range(1, 11)))"),
    ("Write a script to compute 5 factorial", "import math\nprint(math.factorial(5))"),
    ("Write a script to check if 17 is prime",
     "def is_prime(n):\n    if n < 2: return False\n    for i in range(2, int(n**0.5)+1):\n        if n % i == 0: return False\n    return True\nprint(is_prime(17))"),
    ("Write a script to reverse a string", "print('hello'[::-1])"),
    ("Write a script to find the max of a list", "print(max([3, 7, 2, 9, 1]))"),
    ("Write a script to sort a list", "print(sorted([3, 1, 4, 1, 5, 9, 2, 6]))"),
    ("Write a script to count vowels", "print(sum(1 for c in 'education' if c in 'aeiou'))"),
    ("Write a script to check if a string is a palindrome", "print('racecar' == 'racecar'[::-1])"),
    ("Write a script to compute Fibonacci",
     "a, b = 0, 1\nfor _ in range(10):\n    print(a, end=' ')\n    a, b = b, a + b"),
    ("Write a script to compute GCD",
     "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a\nprint(gcd(12, 8))"),
    ("Write a script to print a multiplication table",
     "for i in range(1, 4):\n    for j in range(1, 4):\n        print(i*j, end=' ')\n    print()"),
    ("Write a script to count words in a sentence", "print(len('the quick brown fox'.split()))"),
    ("Write a script to convert string to uppercase", "print('hello'.upper())"),
    ("Write a script to list squares 1 to 10", "print([x**2 for x in range(1, 11)])"),
    ("Write a script to filter even numbers", "print([x for x in range(20) if x % 2 == 0])"),
    ("Write a script to compute factorial", "import math\nprint(math.factorial(10))"),
]
for task, code in scripts:
    examples.append({"messages": [
        {"role": "system", "content": disc_sys},
        {"role": "user", "content": task},
        {"role": "assistant", "content": f"Here is a script.\n{tc('run_script', code=code)}"},
    ]})

# calculate
for a in range(2, 20):
    for b in range(2, 20):
        if random.random() < 0.3:
            examples.append({"messages": [
                {"role": "system", "content": disc_sys},
                {"role": "user", "content": f"What is {a} * {b}?"},
                {"role": "assistant", "content": f"Let me calculate.\n{tc('calculate', code=f'{a} * {b}')}"},
            ]})
        if random.random() < 0.3:
            examples.append({"messages": [
                {"role": "system", "content": disc_sys},
                {"role": "user", "content": f"What is {a} + {b}?"},
                {"role": "assistant", "content": f"Let me calculate.\n{tc('calculate', code=f'{a} + {b}')}"},
            ]})

# web_search
search_queries = ["prime numbers", "fibonacci sequence", "golden ratio", "binary search",
                  "python programming", "machine learning", "quantum computing",
                  "collatz conjecture", "twin primes", "euclidean algorithm"]
for q in search_queries:
    examples.append({"messages": [
        {"role": "system", "content": disc_sys},
        {"role": "user", "content": f"Search for information about {q}."},
        {"role": "assistant", "content": f"Let me search the web.\n{tc('web_search', query=q)}"},
    ]})

# wikipedia_search
wiki_queries = ["Albert Einstein", "Pythagorean theorem", "Fibonacci", "Prime number",
                "Golden ratio", "Binary search algorithm", "Factorial", "Palindrome"]
for q in wiki_queries:
    examples.append({"messages": [
        {"role": "system", "content": disc_sys},
        {"role": "user", "content": f"Tell me about {q}."},
        {"role": "assistant", "content": f"Let me search Wikipedia.\n{tc('wikipedia_search', query=q)}"},
    ]})

# arxiv_search
arxiv_queries = ["transformer architecture", "reinforcement learning", "neural networks",
                 "quantum computing", "graph neural networks"]
for q in arxiv_queries:
    examples.append({"messages": [
        {"role": "system", "content": disc_sys},
        {"role": "user", "content": f"Find papers about {q}."},
        {"role": "assistant", "content": f"Let me search arXiv.\n{tc('arxiv_search', query=q)}"},
    ]})

# fetch_url
examples.append({"messages": [
    {"role": "system", "content": disc_sys},
    {"role": "user", "content": "Read the content of https://en.wikipedia.org/wiki/Prime_number"},
    {"role": "assistant", "content": f"Let me fetch that page.\n{tc('fetch_url', url='https://en.wikipedia.org/wiki/Prime_number')}"},
]})

# save_research
examples.append({"messages": [
    {"role": "system", "content": disc_sys},
    {"role": "user", "content": "Begin exploring."},
    {"role": "assistant", "content": f"I found an interesting article.\n{tc('save_research', query='prime numbers', summary='Primes are numbers divisible only by 1 and themselves.', url='https://en.wikipedia.org/wiki/Prime_number')}"},
]})

# propose_theory
theories = [
    ("The sum of two even numbers is always even.", "Mathematical property."),
    ("The sum of two odd numbers is always even.", "Mathematical property."),
    ("A number divisible by 9 has a digit sum divisible by 9.", "Divisibility rule."),
    ("Prime numbers greater than 5 end in 1, 3, 7, or 9.", "Property of primes."),
    ("The Fibonacci sequence converges to the golden ratio.", "Convergence."),
    ("Every even number greater than 2 is the sum of two primes.", "Goldbach conjecture."),
]
for statement, notes in theories:
    examples.append({"messages": [
        {"role": "system", "content": disc_sys},
        {"role": "user", "content": "Begin exploring."},
        {"role": "assistant", "content": f"I have a hypothesis.\n{tc('propose_theory', statement=statement, notes=notes)}"},
    ]})

# update_theory
examples.append({"messages": [
    {"role": "system", "content": disc_sys},
    {"role": "user", "content": "Begin exploring."},
    {"role": "assistant", "content": f"The theory is confirmed.\n{tc('update_theory', theory_id=1, status='supported', evidence_for=3)}"},
]})

# record_discovery
discoveries = [
    "The first 10 primes are 2, 3, 5, 7, 11, 13, 17, 19, 23, 29.",
    "The factorial of 5 is 120.",
    "The Fibonacci sequence starts 0, 1, 1, 2, 3, 5, 8, 13, 21, 34.",
    "The sum of 1 to 10 is 55.",
    "2^10 = 1024.",
    "17 is a prime number.",
    "The GCD of 12 and 8 is 4.",
    "The golden ratio is approximately 1.618.",
]
for summary in discoveries:
    for conf in [0.9, 0.95]:
        examples.append({"messages": [
            {"role": "system", "content": disc_sys},
            {"role": "user", "content": "Begin exploring."},
            {"role": "assistant", "content": f"I have confirmed a finding.\n{tc('record_discovery', summary=summary, confidence=conf)}"},
        ]})

# query_db
queries = ["SELECT * FROM thoughts LIMIT 5", "SELECT * FROM discoveries LIMIT 5",
           "SELECT * FROM theories WHERE status='supported'", "SELECT COUNT(*) FROM thoughts"]
for sql in queries:
    examples.append({"messages": [
        {"role": "system", "content": disc_sys},
        {"role": "user", "content": "Begin exploring."},
        {"role": "assistant", "content": f"Let me query the database.\n{tc('query_db', sql=sql)}"},
    ]})

# migrate_schema
examples.append({"messages": [
    {"role": "system", "content": disc_sys},
    {"role": "user", "content": "Begin exploring."},
    {"role": "assistant", "content": f"I need to add a table for experiments.\n{tc('migrate_schema', sql='CREATE TABLE experiments (id INTEGER PRIMARY KEY, name TEXT, result TEXT)', reason='Track experiment results')}"},
]})

# summarize_context
examples.append({"messages": [
    {"role": "system", "content": disc_sys},
    {"role": "user", "content": "Begin exploring."},
    {"role": "assistant", "content": f"Let me summarize what I have found so far.\n{tc('summarize_context', summary='I explored prime numbers and found the first 20 primes. I also verified that the sum of two odd numbers is always even.', confidence=0.9)}"},
]})

# finish_session
for phrase in ["I have finished exploring.", "Done for now.", "Session complete."]:
    examples.append({"messages": [
        {"role": "system", "content": disc_sys},
        {"role": "user", "content": "Begin exploring."},
        {"role": "assistant", "content": f"{phrase}\n{tc('finish_session', summary='Explored prime numbers and Fibonacci sequence.')}"},
    ]})

# set_goal
goals = ["Understand prime number distribution", "Explore the Fibonacci sequence",
         "Investigate divisibility rules", "Study sorting algorithms"]
for goal in goals:
    examples.append({"messages": [
        {"role": "system", "content": disc_sys},
        {"role": "user", "content": "Begin exploring."},
        {"role": "assistant", "content": f"I want to set a goal for this session.\n{tc('set_goal', goal=goal)}"},
    ]})

# ask_clarification
examples.append({"messages": [
    {"role": "system", "content": disc_sys},
    {"role": "user", "content": "Tell me about numbers."},
    {"role": "assistant", "content": f"I need more information.\n{tc('ask_clarification', question='What type of numbers are you interested in? Prime numbers, Fibonacci, or something else?')}"},
]})

# ── 3. Multi-turn trajectories with discovery tools ────────────────────────
exploration_topics = [
    ("prime numbers",
     "primes = [n for n in range(2, 50) if all(n % i != 0 for i in range(2, int(n**0.5)+1))]\nprint(primes)",
     "[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]"),
    ("Fibonacci numbers",
     "a, b = 0, 1\nfib = []\nfor _ in range(10):\n    fib.append(a)\n    a, b = b, a + b\nprint(fib)",
     "[0, 1, 1, 2, 3, 5, 8, 13, 21, 34]"),
    ("perfect squares",
     "print([x**2 for x in range(1, 11)])",
     "[1, 4, 9, 16, 25, 36, 49, 64, 81, 100]"),
    ("even numbers",
     "print([x for x in range(20) if x % 2 == 0])",
     "[0, 2, 4, 6, 8, 10, 12, 14, 16, 18]"),
    ("factorials",
     "import math\nprint([math.factorial(i) for i in range(6)])",
     "[1, 1, 2, 6, 24, 120]"),
    ("palindromes",
     "words = ['racecar', 'hello', 'level', 'world', 'noon']\nprint([w for w in words if w == w[::-1]])",
     "['racecar', 'level', 'noon']"),
    ("string reversal",
     "print('python'[::-1])",
     "nohtyp"),
    ("list sorting",
     "print(sorted([5, 2, 8, 1, 9, 3]))",
     "[1, 2, 3, 5, 8, 9]"),
    ("sum 1 to 100",
     "print(sum(range(1, 101)))",
     "5050"),
    ("GCD",
     "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a\nprint(gcd(12, 8))",
     "4"),
]

for topic, code, output in exploration_topics:
    # 3-turn
    examples.append({"messages": [
        {"role": "system", "content": disc_sys},
        {"role": "user", "content": "Begin exploring."},
        {"role": "assistant", "content": f"I want to explore {topic}.\n{tc('run_script', code=code)}"},
        {"role": "tool", "content": tr({"stdout": output, "returncode": 0, "ok": True, "script_id": 1})},
        {"role": "assistant", "content": f"Found {topic}: {output}.\n{tc('record_discovery', summary=f'{topic}: {output}', confidence=0.9)}"},
        {"role": "tool", "content": tr({"discovery_id": 1, "saved": True})},
        {"role": "assistant", "content": f"Done.\n{tc('finish_session', summary=f'Explored {topic}')}"},
    ]})
    # 5-turn with think
    examples.append({"messages": [
        {"role": "system", "content": disc_sys},
        {"role": "user", "content": "Begin exploring."},
        {"role": "assistant", "content": f"Let me think about {topic}.\n{tc('think', content=f'I want to understand {topic} by computing them.', confidence=0.8)}"},
        {"role": "tool", "content": tr({"thought_id": 1, "saved": True})},
        {"role": "assistant", "content": f"Now let me compute {topic}.\n{tc('run_script', code=code)}"},
        {"role": "tool", "content": tr({"stdout": output, "returncode": 0, "ok": True, "script_id": 1})},
        {"role": "assistant", "content": f"Found {topic}: {output}.\n{tc('record_discovery', summary=f'{topic}: {output}', confidence=0.9)}"},
        {"role": "tool", "content": tr({"discovery_id": 1, "saved": True})},
        {"role": "assistant", "content": f"Done.\n{tc('finish_session', summary=f'Explored {topic}')}"},
    ]})

# ── 4. Plain code generation (no tools) ────────────────────────────────────
plain_code = [
    ("Write a Python function to add two numbers", "def add(a, b):\n    return a + b"),
    ("Write a Python function to check if even", "def is_even(n):\n    return n % 2 == 0"),
    ("Write a Python function to square a number", "def square(n):\n    return n ** 2"),
    ("Write a Python function to reverse a string", "def reverse(s):\n    return s[::-1]"),
    ("Write a Python function to check prime",
     "def is_prime(n):\n    if n < 2: return False\n    for i in range(2, int(n**0.5)+1):\n        if n % i == 0: return False\n    return True"),
    ("Write a Python function to compute factorial",
     "def factorial(n):\n    if n <= 1: return 1\n    return n * factorial(n-1)"),
    ("Write a Python function to compute fibonacci",
     "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n): a, b = b, a+b\n    return a"),
    ("Write a Python function to count vowels",
     "def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')"),
    ("Write a Python function to check palindrome",
     "def is_palindrome(s):\n    return s == s[::-1]"),
    ("Write a Python function to find GCD",
     "def gcd(a, b):\n    while b: a, b = b, a % b\n    return a"),
    ("Write a Python one-liner to print Hello World", "print('Hello World')"),
    ("Write a Python one-liner to sum 1 to 10", "print(sum(range(1, 11)))"),
    ("Write a Python one-liner to reverse a string", "print('hello'[::-1])"),
    ("Write a Python one-liner to create squares", "print([x**2 for x in range(1, 6)])"),
    ("Write a Python one-liner to sort a list", "print(sorted([3, 1, 4, 1, 5]))"),
    ("Write a Python script to print numbers 1 to 5", "for i in range(1, 6):\n    print(i)"),
    ("Write a Python script to print even numbers", "for i in range(2, 11, 2):\n    print(i)"),
    ("Write a Python script to compute Fibonacci",
     "a, b = 0, 1\nfor _ in range(10):\n    print(a, end=' ')\n    a, b = b, a + b"),
]
for prompt, code in plain_code:
    examples.append({"prompt": prompt, "response": code})

# ── Save ───────────────────────────────────────────────────────────────────
random.shuffle(examples)
with open(OUT, "w", encoding="utf-8") as f:
    for ex in examples:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

n_multi = sum(1 for ex in examples if "messages" in ex)
n_single = sum(1 for ex in examples if "prompt" in ex)
print(f"\nTotal: {len(examples)} examples -> {OUT}")
print(f"  Multi-turn (with tools): {n_multi}")
print(f"  Single-turn (plain code): {n_single}")
