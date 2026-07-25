"""(Phase -1) Benchmark problem sets in one unified format.

A `Problem` is a dict:
    id          : str
    prompt      : str   # shown to the model (a function header + docstring)
    test        : str   # defines `check(candidate)` that asserts correctness
    entry_point : str   # the function name `check` will call

So every problem is scored the same way (HumanEval convention): run
`prompt + completion + test + check(entry_point)` in the sandbox.

Sources:
  * MINI_BENCH  — 6 built-in problems, ZERO downloads (offline Gate check + a
    held-out internal set, as the ROADMAP asks for).
  * load_humaneval() / load_mbpp() — the standard HF benchmarks.
"""

# ---------------------------------------------------------------------------
# Built-in offline mini-benchmark (held-out internal set)
# ---------------------------------------------------------------------------
MINI_BENCH = [
    {
        "id": "mini/add",
        "prompt": 'def add(a, b):\n    """Return the sum of a and b."""\n',
        "entry_point": "add",
        "test": ("def check(candidate):\n"
                 "    assert candidate(2, 3) == 5\n"
                 "    assert candidate(-1, 1) == 0\n"
                 "    assert candidate(0, 0) == 0\n"),
    },
    {
        "id": "mini/reverse",
        "prompt": ('def reverse_string(s):\n'
                   '    """Return the string s reversed."""\n'),
        "entry_point": "reverse_string",
        "test": ("def check(candidate):\n"
                 "    assert candidate('abc') == 'cba'\n"
                 "    assert candidate('') == ''\n"
                 "    assert candidate('a') == 'a'\n"),
    },
    {
        "id": "mini/is_even",
        "prompt": ('def is_even(n):\n'
                   '    """Return True if n is even, else False."""\n'),
        "entry_point": "is_even",
        "test": ("def check(candidate):\n"
                 "    assert candidate(2) is True\n"
                 "    assert candidate(3) is False\n"
                 "    assert candidate(0) is True\n"),
    },
    {
        "id": "mini/max_of_list",
        "prompt": ('def max_of_list(xs):\n'
                   '    """Return the largest number in the non-empty list xs."""\n'),
        "entry_point": "max_of_list",
        "test": ("def check(candidate):\n"
                 "    assert candidate([1, 2, 3]) == 3\n"
                 "    assert candidate([-5, -2, -9]) == -2\n"
                 "    assert candidate([7]) == 7\n"),
    },
    {
        "id": "mini/factorial",
        "prompt": ('def factorial(n):\n'
                   '    """Return n! for n >= 0."""\n'),
        "entry_point": "factorial",
        "test": ("def check(candidate):\n"
                 "    assert candidate(0) == 1\n"
                 "    assert candidate(1) == 1\n"
                 "    assert candidate(5) == 120\n"),
    },
    {
        "id": "mini/count_vowels",
        "prompt": ('def count_vowels(s):\n'
                   '    """Return the number of vowels (aeiou) in lowercase string s."""\n'),
        "entry_point": "count_vowels",
        "test": ("def check(candidate):\n"
                 "    assert candidate('hello') == 2\n"
                 "    assert candidate('xyz') == 0\n"
                 "    assert candidate('aeiou') == 5\n"),
    },
]


def load_humaneval(limit: int | None = None) -> list[dict]:
    """OpenAI HumanEval (164 problems). Needs `datasets` + network/cache."""
    from datasets import load_dataset
    ds = load_dataset("openai_humaneval", split="test")
    out = []
    for ex in ds:
        out.append({
            "id": ex["task_id"],
            "prompt": ex["prompt"],
            "test": ex["test"],
            "entry_point": ex["entry_point"],
        })
        if limit and len(out) >= limit:
            break
    return out


def load_mbpp(limit: int | None = None, split: str = "test") -> list[dict]:
    """MBPP. Its tests are a list of asserts referencing a function name; we
    wrap them into a `check(candidate)` that ignores `candidate` and runs the
    asserts against the function the completion defined (MBPP convention)."""
    from datasets import load_dataset
    ds = load_dataset("mbpp", split=split)
    out = []
    for ex in ds:
        asserts = "\n".join("    " + t for t in ex["test_list"])
        test = "def check(candidate):\n" + asserts + "\n"
        # MBPP prompt: NL description as a docstring; the model writes the func.
        prompt = f'"""{ex["text"].strip()}"""\n'
        # entry_point is unused by the asserts but required by the runner shape.
        out.append({
            "id": f"mbpp/{ex['task_id']}",
            "prompt": prompt,
            "test": test,
            "entry_point": "candidate",
        })
        if limit and len(out) >= limit:
            break
    return out


def get_benchmark(name: str, limit: int | None = None) -> list[dict]:
    if name == "mini":
        return MINI_BENCH[:limit] if limit else list(MINI_BENCH)
    if name == "humaneval":
        return load_humaneval(limit)
    if name == "mbpp":
        return load_mbpp(limit)
    raise ValueError(f"unknown benchmark {name!r}; choose mini|humaneval|mbpp")
