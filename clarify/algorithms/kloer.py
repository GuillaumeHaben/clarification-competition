# SPDX-FileCopyrightText: 2026 Guillaume Haben <guillaume.haben@telecomnancy.net>
#
# SPDX-License-Identifier: MIT

"""Kloer.

Analyse, explore, ask, repair, implement, verify.

1. Analyse. A requirements-analyst call, decoupled from code generation, reviews the task
   against an ambiguity checklist (contradictions plus the ambiguity taxonomy of
   ClarifyCodeBench, Fang et al. 2026), picks the open point most likely to break hidden tests,
   drafts a question about it and writes two or three explicit readings of the task.
2. Explore. One candidate implementation is written per reading, rather than by re-sampling,
   because sampling tends to repeat one interpretation (semantic collapse, Richter & Papadakis
   2026). The candidates run on probe inputs written by the model (inputs only) in a single
   sandboxed execution, and are grouped by behaviour (VClare sim-level repair, Zhao et al. 2026;
   output inconsistency as an ambiguity signal, ClarifyGPT, Mu et al. 2024).
3. Ask. One atomic question. If the task contains a quoted contradiction, the question quotes
   both statements and asks which one holds (VClare spec-level repair). Otherwise, if two
   behaviour groups differ on an ordinary input, the question states the behavioural difference
   and illustrates it with that input and the two results (test-driven clarification, TiCoder,
   Lahiri et al. 2022). Otherwise the analyst's question is used.
4. Repair. The task is rewritten into a structured specification, editing only what the answer
   invalidates (VClare targeted repair) and laid out as redundant layers (Akli et al. 2026), with
   meaning-preserving IF-THEN-EXCEPT rules and function-name neutralisation (DualFix, Akli et
   al. 2026). When the candidates disagree on the number of arguments of a task without a
   signature, the extra parameters are made optional so that both call forms work.
5. Implement and verify. The coder works from the specification. The result is executed on the
   specification's examples; if it fails, a candidate that passes them all is preferred, else one
   self-debugging round uses the execution feedback (Chen et al. 2024). The code is sanitised
   deterministically and the neutral name is bound back to the required entry point.
6. Finalise. Deterministic robustness for the evaluation harness: `Counter` is imported (some
   expected outputs are Counter objects) and a lower-case alias of the entry point is defined.
   For tasks without a signature, a call whose arguments do not fit the function is adapted
   instead of raising TypeError: a redundant length argument is dropped, arguments are reordered
   to match type annotations, and a missing length is filled in. Input checks that raise
   exceptions nobody asked for are removed. Only calls that would otherwise fail behave
   differently.

0.4 adds: call-form inference for tasks without a signature (the coder makes every plausible call
form work, since the interface itself is ambiguous: the "Inputs" type of the taxonomy), no
"raise an error" alternatives in questions unless the task mentions errors, substantive
differences preferred over a string-versus-list-of-parts difference, a rewrite rule for examples
that contradict the answer by a constant pattern, and a time guard that skips optional steps when
the task's time limit gets close.

Team: Kloer
Team Members: Milos Ojdanic, Guillaume Haben, Sami Lazreg
Main Contact: guillaume.haben@telecomnancy.net
"""

from __future__ import annotations

import ast
import builtins
import hashlib
import json
import keyword
import os
import re
import time
from typing import Any

from clarify.baselines import ClarificationAlgorithmBase
from clarify.env import (
    ClarificationEnvironment,
    LimitsExceededException,
    TooManyQuestionException,
)
from clarify.runtime import _validate_and_parse_evalplus_result

KLOER_VERSION = "0.4"

NEUTRAL_NAME = "solution"

ANALYST_PROMPT = """
You are a requirements analyst reviewing a coding task before any code is written.
The task asks for a Python function named {entry_point}.

### Task
{prompt}

### Previous clarifications
{clarifications}

### Checklist
Look for points where two reasonable developers could write versions of {entry_point} that
return different outputs for the same input. Check each type below; the last four are the
easiest to overlook, so check them with particular care.
- Contradiction: two statements, or a statement and an example, that cannot both hold.
- Terminology: a domain term, action or state that is undefined or has several meanings.
- Behavior: the main computation or goal is not uniquely determined.
- Output format: the exact return value, type, structure or wording (for example True/False
  versus a message string, a list versus a tuple).
- Inputs: which arguments the function takes, their types and their order.
- Edge cases: empty input, nothing matching, special or invalid values.
- Indices and ranges: 0- or 1-based positions, inclusive or exclusive bounds.
- Units and precision: unit or scale of quantities, rounding, tolerance.
- Comparison rules: the sort or comparison key, tie-breaking, what counts as equal.
- Collection semantics: duplicates, membership, nesting, whether an input is modified in place.
- Ordering: the order of elements in the result, or the order in which steps are applied.

### Instructions
1. List the points you find (at most five). For each, give its type and quote the statement it
   comes from (source_a). For a contradiction, also quote the conflicting statement or example
   (source_b); otherwise leave source_b empty.
2. For each point, decide whether it can be resolved confidently from the examples, the function
   name and signature, the previous clarifications, or a strong common convention, and write the
   resolution you would use as an assumption.
3. Pick the ONE unresolved point most likely to make hidden tests fail if guessed wrong. Prefer,
   in this order: contradictions; the main behavior and output format on ordinary inputs; the
   other types; rare edge cases.
4. Write one clarifying question about that point:
   - For a contradiction, quote the two conflicting statements briefly and ask which one the
     function should follow, e.g.: The task says "..." but also "...". Which one should the
     function follow?
   - Otherwise, offer the two most plausible concrete alternatives, e.g. "Should X return A or B
     when ...?". When it helps, use a small concrete input and its two possible outputs.
   - Ask for exactly one fact about the required behavior (not the implementation or the tests),
     with a single question mark. Never ask a yes/no question that only confirms the obvious
     reading. Make it self-contained, answerable in one sentence, and plain text without
     backticks or code formatting.{no_raise_rule}
5. Write the two or three most plausible complete readings of the task, one sentence each, most
   likely first. They must lead to different outputs for some input, preferably on the point you
   ask about. If only one reading is sensible, still give the closest alternative.
Keep the analysis short.

Answer with a JSON object only:
{"points": [{"type": "...", "source_a": "...", "source_b": "", "resolvable": false, "assumption": "..."}], "interpretations": ["...", "..."], "question": "..."}
""".strip()

FALLBACK_QUESTION_PROMPT = """
A user asked for a Python function named {entry_point}:

{prompt}

Write the single most important clarifying question whose answer would change what
{entry_point} returns. Ask for exactly one fact, offer the two most plausible concrete
alternatives, and do not ask about implementation details or test cases.{no_raise_rule}
Reply with the question only, as plain text without backticks.
""".strip()

NO_RAISE_ANALYST_RULE = (
    "\n   - Do not offer raising an exception (or returning an error message) as one of the"
    "\n     alternatives unless the task itself mentions exceptions or errors; offer two concrete"
    "\n     results instead."
)

NO_RAISE_FALLBACK_RULE = (
    "\nDo not offer raising an exception as an alternative unless the task mentions exceptions or"
    "\nerrors; offer two concrete results instead."
)

REPAIR_PROMPT = """
Rewrite the coding task below into a clear, structured specification of a Python function.
A developer will implement the function from your specification alone, without seeing the
original task.

### Original task
{prompt}

### Clarifications from the user (authoritative)
{clarifications}

### Assumptions for points the user did not clarify
{assumptions}

### How to rewrite
- Repair, do not reinvent. Change only the statements that a clarification contradicts or makes
  more precise, and keep every other piece of information. Keep every example whose result agrees
  with the clarifications; drop an example only if it contradicts a clarification, and never
  compute new expected results. Check every example against the clarifications, including their
  indirect consequences (e.g. if the user says 1 is not prime, an example whose result contains 1
  is wrong), and drop each example that the clarifications make wrong.
- Write results as Python literals; a tuple with one element needs a trailing comma, e.g.
  ('Venus',).
- If the task contradicts itself and no clarification settles it, keep the reading supported by
  the examples, then by the function name, then the most common reading, and state only that one.
- Do not add requirements that appear in none of the sources above (no extra input validation,
  error handling or performance constraints). Use an assumption only where the task and the
  clarifications are silent.
- Apply these meaning-preserving rules wherever they match:
  R1. IF a precise algorithmic or mathematical term appears, THEN add a short clarifying
      parenthetical, EXCEPT when the phrase denotes a standard relation or order.
  R2. IF a formula or constraint uses mathematical symbols or notation, THEN also state it in
      plain English, keeping the exact comparators, limits and formula.
  R3. IF the function checks, tests or confirms whether something holds, THEN state exactly what
      it returns in each case (e.g. "Return True if ..., otherwise return False"), EXCEPT keep any
      other return values that the task or its examples use (e.g. the strings "Valid"/"Invalid").
  R4. IF there are typos, stray symbols or broken formatting, THEN fix them without changing the
      meaning.
- {naming_rule}{extra_rules}

Write the specification as plain text, in exactly this format:
SIGNATURE: def {name}(<parameters>)
DESCRIPTION: <what the function computes, in one or two sentences>
INPUTS: <each parameter: its type and meaning>
OUTPUT: <the exact return value and its type>
RULES:
- <one line per behavior rule, edge case or clarified point>
EXAMPLES:
- <call> -> <expected result>
HELPER CODE:
<other functions or constants given in the task, copied verbatim, or (none)>
""".strip()

NAMING_RULE_NEUTRAL = (
    "Call the function {name} in the SIGNATURE and in the examples, with exactly the parameters "
    "of the task's signature. The original name {entry_point} must not appear in the "
    "specification: if it conveys behavior that the task does not state explicitly, state that "
    "behavior in the DESCRIPTION instead."
)

NAMING_RULE_KEEP = (
    "Keep the function name {name}. If the task gives a signature, copy its parameters exactly; "
    "otherwise use the parameter list that the task and the clarifications imply."
)

SPEC_CODER_PROMPT = """
Implement the Python function `{name}` described by the specification below.

### Specification
{spec}

### Clarifications from the user
{clarifications}
{original}
### Rules
- Follow the specification and the clarifications exactly; if they disagree, the clarifications
  win.
- Use exactly the function name `{name}` and the parameters of the SIGNATURE line. Include the
  helper code if your function needs it.
- Do not add input validation, assertions or exceptions that the specification does not require.{extra_rules}
- Use only the Python standard library. Do not read input, print, or include example usage or
  tests.

Return the complete, self-contained solution in a single ```python block.
""".strip()

ORIGINAL_SECTION = """
### Original task (for reference only; the specification takes precedence)
{prompt}
"""

CODER_PROMPT = """
Implement the Python function `{entry_point}` for the task below.

### Task
{prompt}

### Clarifications from the user
{clarifications}

### Assumptions for points that were not clarified
{assumptions}

### Rules
- The user's clarifications are authoritative and override anything in the task they conflict with.
- When the task contradicts itself, prefer, in this order: the clarifications, the examples or
  doctests, the function name and signature, and finally the most common reading of the task.
- Do not follow requirements that clearly conflict with the examples or the function's purpose
  (unusual return types, artificial restrictions) unless a clarification confirms them.
- Keep the exact function name `{entry_point}`. If no signature is given, choose the most natural
  parameter list for the described inputs.
- Use only the Python standard library. Do not read input, print, or include example usage.

Return the complete, self-contained solution in a single ```python block.
""".strip()

FORMAT_REMINDER = (
    "Your answer must contain the full implementation of `{name}` in a single "
    "```python block. Please answer again."
)

SPEC_SECTIONS = ("SIGNATURE:", "DESCRIPTION:", "OUTPUT:")

CANDIDATE_PROMPT = """
Implement the Python function `{entry_point}` for the task below, following the given reading of
the task exactly.

### Task
{prompt}

### Reading to implement
{reading}

### Rules
- Where the reading and the task disagree, follow the reading.
- Keep the exact function name `{entry_point}`. If the task gives a signature, keep it; otherwise
  choose the most natural parameter list for the described inputs.
- Do not add input validation, assertions or exceptions that the task does not require.
- Use only the Python standard library. Do not read input, print, or include example usage or
  tests.

Return the complete solution in a single ```python block.
""".strip()

PROBE_PROMPT = """
Write {n} test inputs for the Python function described below. They will be used to compare
different implementations, so each must be a valid input that the task allows.

### Task
{prompt}

### Signature
{signature}

### Instructions
- Start with small, ordinary inputs that exercise the main behaviour, then add a few edge cases
  (empty or single-element inputs, ties, boundaries).
- Write each input on its own line as a Python list of the positional arguments, e.g.
  [[3, 1, 2], 2] for a call with the arguments [3, 1, 2] and 2, or ["abc"] for a call with the
  single argument "abc".
- Use only literals (numbers, strings, lists, tuples, dicts, sets, booleans, None) and keep each
  input short.
- Do not write expected outputs or any explanation.

Answer with the inputs only, inside a single ```text block.
""".strip()

VALUE_QUESTION_PROMPT = """
Two plausible readings of a coding task give different results for the same call.

### Task
{prompt}

### The difference
Reading A: {reading_a}
With reading A, {call} returns {value_a}.
Reading B: {reading_b}
With reading B, {call} returns {value_b}.

Write ONE clarifying question for the user that asks which behaviour the function must have.
State the behavioural difference in plain words and illustrate it with the call and the two
results, for example: Should the function ... (so that {call} returns {value_a}) or ... (so that
it returns {value_b})?
Ask for exactly one fact, with a single question mark, in at most 60 words. Do not mention
readings, candidates or implementations. Write plain text without backticks.
Reply with the question only.
""".strip()

GENERIC_QUESTION = (
    "What is the most important rule that {entry_point} must follow to compute its result?"
)

VALUE_QUESTION_TEMPLATE = "For {call}, should the result be {value_a} or {value_b}?"

ARITY_QUESTION_TEMPLATE = (
    "Should {entry_point} take the arguments ({short}) or the arguments ({long})?"
)

ARITY_RULE = (
    "Callers may pass only the first {k} of the arguments ({params}) or all {m} of them. Accept "
    "both: give the parameters after the first {k} default values such that leaving them out "
    "gives the same result."
)

OFFERED_RULE = (
    "The question offered two results for {call}: {value_a} or {value_b}. If the answer settles "
    "which result is right (or gives another one), add that call with the right result to "
    "EXAMPLES."
)

EXAMPLE_PATTERN_RULE = (
    "If the clarifications describe a behavior that the task's examples all contradict by the same "
    "pattern (for example every expected result is ten times, or one more than, what the described "
    "behavior gives, or has one extra element), those examples are wrong: drop them."
)

CALL_FORMS_PROMPT = """
A Python function named {entry_point} must be written for the task below. The task does not give
its signature, and the function will be called in one fixed way that the task does not state.

### Task
{prompt}

### Clarifications from the user
{clarifications}

List the two or three most plausible ways to call {entry_point}, most plausible first. They must
differ in the number, order, grouping or types of the positional arguments. Consider whether:
- a compound value (a point, a pair, a complex number, a date, a range) is passed as one argument
  or as separate scalar arguments;
- several values are passed as separate arguments or together in one list or tuple;
- an input is a collection of items or a single number (such as a count or a limit);
- a parameter of the computation (such as a threshold or a set of values to look for) is passed
  as an argument or fixed inside the function.
Do not list forms that differ only by an extra argument giving the length of a collection: such
calls are handled automatically.

Write each form as a Python signature with type annotations, one per line, for example:
def distance(start: tuple[float, float], end: tuple[float, float])
def distance(x1: float, y1: float, x2: float, y2: float)

Answer with the signatures only, inside a single ```text block.
""".strip()

CALL_FORMS_RULE = (
    "The task does not say how {entry_point} is called, and the caller may use any of these forms: "
    "{forms}. Write one function {entry_point} that accepts every form and gives the same result "
    "for the same data, for example def {entry_point}(*args) that checks the number and the types "
    "of its arguments and converts them to the first form."
)

DEBUG_PROMPT = """
Your implementation of `{name}` does not match its specification on these examples (call,
expected result, what your code did):
{failures}

### Specification
{spec}

### Clarifications from the user
{clarifications}

### Your implementation
```python
{code}
```

Fix the implementation so that it satisfies the specification and its examples. If an example
contradicts a clarification from the user, follow the clarification. Keep the function name
`{name}` and its parameters. Do not add input validation or exceptions that the specification
does not require.

Return the complete corrected solution in a single ```python block.
""".strip()

# Runs every candidate on every input in ONE sandboxed process (the SDK starts one container per
# exec_code call) and prints one JSON row per candidate. Each call has a time limit; the whole run
# has a time budget below the sandbox's own 30 s timeout. Rows are flushed one by one, so a run
# that is killed still reports the candidates it finished.
HARNESS = r"""
import contextlib, copy, io, json, math, signal, sys, time
sys.setrecursionlimit(10000)
inf, nan = float("inf"), float("nan")
CODES = __CODES__
INPUTS = __INPUTS__
EXPECTED = __EXPECTED__
ENTRY = __ENTRY__
LIMIT = __LIMIT__
DEADLINE = time.monotonic() + __BUDGET__


class KloerLate(BaseException):
    pass


def kloer_alarm(signum, frame):
    raise KloerLate()


signal.signal(signal.SIGALRM, kloer_alarm)


def kloer_norm(value):
    if isinstance(value, float):
        return value if math.isnan(value) or math.isinf(value) else float(f"{value:.9g}")
    if isinstance(value, list):
        return [kloer_norm(x) for x in value]
    if isinstance(value, tuple):
        return tuple(kloer_norm(x) for x in value)
    if isinstance(value, dict):
        return {k: kloer_norm(x) for k, x in value.items()}
    return value


def kloer_same(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b or math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(map(kloer_same, a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(kloer_same(a[k], b[k]) for k in a)
    try:
        return bool(a == b)
    except Exception:
        return False


def kloer_limited(function, *args):
    signal.setitimer(signal.ITIMER_REAL, LIMIT)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return function(*args)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


for index, code in enumerate(CODES):
    try:
        namespace = {"__name__": "kloer_candidate"}
        kloer_limited(exec, code, namespace)
        function = namespace[ENTRY]
    except BaseException as exc:
        row = [["err", "load:" + type(exc).__name__]] * len(INPUTS)
        print("KLOER_ROW", index, json.dumps(row), flush=True)
        continue
    row = []
    for position, args in enumerate(INPUTS):
        if time.monotonic() > DEADLINE:
            row.append(["err", "skipped"])
            continue
        try:
            value = kloer_limited(function, *copy.deepcopy(args))
        except BaseException as exc:
            row.append(["err", type(exc).__name__])
            continue
        text = repr(kloer_norm(value))[:200]
        if EXPECTED is None:
            row.append(["ok", text])
        else:
            row.append(["pass" if kloer_same(value, EXPECTED[position]) else "fail", text])
    print("KLOER_ROW", index, json.dumps(row), flush=True)
"""


# Question hygiene ---------------------------------------------------------------------------


def clean_question(question: str) -> str:
    """Makes a question safe for the simulated user.

    The SDK parses the simulated user's reply with ``ANSWERS=`(.+?)` ``, so an answer that echoes
    a backtick from the question is cut short. Backticks are therefore removed. Only the first
    question is kept if several were produced (atomicity).
    """
    question = question.replace("`", "").strip().strip('"').strip()
    question = re.sub(r"\s+", " ", question)
    if question.count("?") > 1:
        question = question[: question.index("?") + 1]
    return question


# Specification repair -----------------------------------------------------------------------


def has_signature(prompt: str, entry_point: str) -> bool:
    """True if the task itself writes the entry point's signature (HumanEval style)."""
    return re.search(rf"\bdef\s+{re.escape(entry_point)}\s*\(", prompt) is not None


def clean_spec(text: str, entry_point: str, name: str) -> str | None:
    """Validates a repaired specification and enforces the neutral name; None if unusable."""
    text = re.sub(r"^```[^\n]*\n|\n?```\s*$", "", text.strip()).strip()
    start = text.find("SIGNATURE:")
    if start < 0:
        return None
    text = text[start:]
    if not all(section in text for section in SPEC_SECTIONS):
        return None
    if name != entry_point:
        # Calls to the original name are renamed; prose mentioning it is left alone.
        text = re.sub(rf"\b{re.escape(entry_point)}(\s*\()", name + r"\1", text)
        if f"{name}(" not in text.split("\n", 1)[0]:
            return None
    return text


# Code extraction and sanitisation ---------------------------------------------------------------


def parse_json_object(text: str) -> dict[str, Any]:
    """Extracts the first JSON object from an LLM response (tolerates code fences)."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def top_level_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def is_main_guard(node: ast.stmt) -> bool:
    return isinstance(node, ast.If) and "__main__" in ast.unparse(node.test)


def calls_input(node: ast.stmt) -> bool:
    return any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "input"
        for n in ast.walk(node)
    )


def is_demo_statement(node: ast.stmt, local_functions: set[str]) -> bool:
    """Top-level code that only demonstrates or tests the solution."""
    if is_main_guard(node) or isinstance(node, ast.Assert) or calls_input(node):
        return True
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
        func = node.value.func
        name = func.id if isinstance(func, ast.Name) else None
        return name == "print" or name in local_functions
    return False


def sanitize_code(code: str, entry_point: str, generated_name: str | None = None) -> str | None:
    """Removes demo code and binds the entry point; returns None if it cannot be bound.

    `generated_name` is the (neutral) name the coder was asked to use, if different.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    local_functions = {
        n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    body = [n for n in tree.body if not is_demo_statement(n, local_functions)]
    changed = len(body) != len(tree.body)
    tree.body = body

    if entry_point not in top_level_names(tree):
        functions = [n.name for n in body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        same_name = [f for f in functions if f.lower() == entry_point.lower()]
        if generated_name and generated_name in functions:
            target = generated_name
        elif same_name:
            target = same_name[0]
        else:
            target = functions[0] if len(functions) == 1 else None
        if target is None:
            return None
        tree.body.append(ast.parse(f"{entry_point} = {target}").body[0])
        changed = True

    return ast.unparse(tree) if changed else code


def extract_code(response: str, entry_point: str, generated_name: str | None = None) -> str | None:
    """Returns sanitised code defining `entry_point`, tolerating fence variants such as
    "``` python", "```py" or an untagged ``` block (common with local models)."""
    normalized = re.sub(r"```[ \t]*(?:python3?|py)\b", "```python", response, flags=re.IGNORECASE)
    candidates = []
    try:
        candidates.append(_validate_and_parse_evalplus_result(normalized))
    except ValueError:
        pass
    candidates += re.findall(r"```[^\n]*\n(.*?)```", normalized, flags=re.DOTALL)
    candidates.append(response)  # bare code without any fence
    for block in candidates:
        code = sanitize_code(block, entry_point, generated_name)
        if code is not None:
            return code
    return None


# Exploration helpers ------------------------------------------------------------------------


def entry_function(tree: ast.Module, entry_point: str) -> ast.FunctionDef | None:
    """The function definition bound to `entry_point` (directly or through an alias)."""
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    if entry_point in functions:
        return functions[entry_point]
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == entry_point for t in node.targets)
            and isinstance(node.value, ast.Name)
        ):
            return functions.get(node.value.id)
    return None


def entry_params(code: str, entry_point: str) -> list[str] | None:
    """Positional parameter names of the entry point, or None if unknown (or *args)."""
    try:
        function = entry_function(ast.parse(code), entry_point)
    except SyntaxError:
        return None
    if function is None or function.args.vararg is not None:
        return None
    return [a.arg for a in function.args.posonlyargs + function.args.args]


def required_count(code: str, entry_point: str) -> int | None:
    try:
        function = entry_function(ast.parse(code), entry_point)
    except SyntaxError:
        return None
    if function is None:
        return None
    positional = function.args.posonlyargs + function.args.args
    return len(positional) - len(function.args.defaults)


def signature_line(code: str, entry_point: str) -> str:
    try:
        function = entry_function(ast.parse(code), entry_point)
    except SyntaxError:
        function = None
    if function is None:
        return f"def {entry_point}(...)"
    return f"def {entry_point}({ast.unparse(function.args)})"


def literal_arguments(node: ast.Call) -> list[Any] | None:
    if node.keywords:
        return None
    try:
        return [ast.literal_eval(arg) for arg in node.args]
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return None


def parse_probe_inputs(
    text: str, entry_point: str, required: int, total: int, limit: int
) -> list[list[Any]]:
    """Argument lists from the probe response (one Python literal or call per line)."""
    blocks = re.findall(r"```[^\n]*\n(.*?)```", text, flags=re.DOTALL)
    lines = (blocks[0] if blocks else text).splitlines()
    inputs: list[list[Any]] = []
    seen: set[str] = set()
    for line in lines:
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip().rstrip(",")
        if not line:
            continue
        try:
            node = ast.parse(line, mode="eval").body
        except SyntaxError:
            continue
        if isinstance(node, ast.Call):
            args = literal_arguments(node)
        else:
            try:
                value = ast.literal_eval(node)
            except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
                continue
            if isinstance(value, (list, tuple)) and required <= len(value) <= total:
                args = list(value)
            elif total == 1:
                args = [value]
            else:
                args = None
        if args is None or not required <= len(args) <= total:
            continue
        key = repr(args)
        if key not in seen and len(key) <= 400:
            seen.add(key)
            inputs.append(args)
        if len(inputs) >= limit:
            break
    return inputs


def parse_call_forms(text: str, entry_point: str, limit: int = 3) -> list[str]:
    """Distinct call forms ("f(a: int, b: list[int])") from the call-forms response.

    Forms are compared by the number and annotations of their parameters, so forms that differ
    only in parameter names count once. Unparsable lines are ignored.
    """
    blocks = re.findall(r"```[^\n]*\n(.*?)```", text, flags=re.DOTALL)
    lines = (blocks[0] if blocks else text).splitlines()
    forms: list[str] = []
    seen: set[tuple[str, ...]] = set()
    for line in lines:
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip().rstrip(":").strip()
        line = re.sub(r"\s*->.*$", "", line)
        if not line.startswith("def "):
            line = "def " + line
        try:
            node = ast.parse(line + ": pass").body[0]
        except SyntaxError:
            continue
        if not isinstance(node, ast.FunctionDef) or node.args.vararg or node.args.kwonlyargs:
            continue
        params = node.args.posonlyargs + node.args.args
        if not params:
            continue
        key = tuple(ast.unparse(p.annotation) if p.annotation else "?" for p in params)
        if key in seen:
            continue
        seen.add(key)
        forms.append(f"{entry_point}({ast.unparse(node.args)})")
        if len(forms) >= limit:
            break
    return forms


def build_harness(
    codes: list[str],
    entry_point: str,
    inputs: list[list[Any]],
    expected: list[Any] | None,
    limit_s: float,
    budget_s: float = 20.0,
) -> str:
    return (
        HARNESS.replace("__EXPECTED__", repr(expected))
        .replace("__ENTRY__", repr(entry_point))
        .replace("__LIMIT__", repr(float(limit_s)))
        .replace("__BUDGET__", repr(float(budget_s)))
        .replace("__INPUTS__", repr(inputs))
        .replace("__CODES__", repr(codes))
    )


def parse_harness_output(output: str, count: int) -> list[list[list[str]] | None]:
    rows: list[list[list[str]] | None] = [None] * count
    for line in output.splitlines():
        if not line.startswith("KLOER_ROW "):
            continue
        parts = line.split(" ", 2)
        try:
            index, row = int(parts[1]), json.loads(parts[2])
        except (IndexError, ValueError):
            continue
        if 0 <= index < count and isinstance(row, list):
            rows[index] = row
    return rows


def canonical(text: str) -> Any:
    """Value of a result repr with representation details removed (tuple/list, int/float,
    single-element wrapping); the repr itself if it is not a literal."""

    def norm(value: Any) -> Any:
        if isinstance(value, bool) or value is None or isinstance(value, str):
            return value
        if isinstance(value, (int, float)):
            return round(float(value), 9)
        if isinstance(value, (list, tuple)):
            items = [norm(v) for v in value]
            return items[0] if len(items) == 1 else ("seq", tuple(map(repr, items)))
        if isinstance(value, (set, frozenset)):
            return ("set", tuple(sorted(map(repr, map(norm, value)))))
        if isinstance(value, dict):
            return ("dict", tuple(sorted((repr(norm(k)), repr(norm(v))) for k, v in value.items())))
        return repr(value)

    try:
        return norm(ast.literal_eval(text))
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return text


def structure_only(text_a: str, text_b: str) -> bool:
    """True if two results hold the same text and differ only in grouping: a string versus the
    list (or tuple) of its characters or words, e.g. 'ab cd' versus ['ab', 'cd']."""
    try:
        values = (ast.literal_eval(text_a), ast.literal_eval(text_b))
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return False
    for text, parts in (values, values[::-1]):
        if (
            isinstance(text, str)
            and isinstance(parts, (list, tuple))
            and all(isinstance(p, str) for p in parts)
        ):
            parts = list(parts)
            joined = {"".join(parts), " ".join(parts), ",".join(parts), ", ".join(parts)}
            return text in joined or text.split() == parts
    return False


def distinguishing_input(
    rows: list[list[list[str]] | None], max_len: int = 60, structure_last: bool = False
) -> tuple[int, int, int, int] | None:
    """(input index, candidate a, candidate b, number of behaviour groups) or None.

    Candidate a is the one that runs on most inputs (ties: the most likely reading, listed
    first); b is the first candidate whose result differs from a's on an input where both return
    a short value that differs in substance, not only in representation (a tuple instead of a
    list, 2 instead of 2.0, [x] instead of x). Inputs are tried in order, ordinary inputs first.
    With structure_last, a difference that is only a string versus the list of its parts is used
    only if no other difference exists.
    """
    present = [i for i, row in enumerate(rows) if row and any(c[0] == "ok" for c in row)]
    if len(present) < 2:
        return None
    groups = {tuple(map(tuple, rows[i])) for i in present}
    a = max(present, key=lambda i: (sum(c[0] == "ok" for c in rows[i]), -i))
    passes = (False, True) if structure_last else (None,)
    for allow_structure in passes:
        for position in range(len(rows[a])):
            cell_a = rows[a][position]
            if cell_a[0] != "ok" or len(cell_a[1]) > max_len:
                continue
            for b in present:
                cell_b = rows[b][position] if position < len(rows[b]) else None
                if b == a or not cell_b or cell_b[0] != "ok" or len(cell_b[1]) > max_len:
                    continue
                if canonical(cell_b[1]) == canonical(cell_a[1]):
                    continue
                if allow_structure is None or allow_structure == structure_only(
                    cell_a[1], cell_b[1]
                ):
                    return position, a, b, len(groups)
    return None


def parse_spec_examples(spec: str, names: set[str], limit: int = 10) -> list[tuple[list[Any], Any]]:
    """(arguments, expected result) pairs from the EXAMPLES section of a specification."""
    match = re.search(r"EXAMPLES:[ \t]*\n(.*?)(?:\nHELPER CODE:|\Z)", spec, flags=re.DOTALL)
    if not match:
        return []
    examples: list[tuple[list[Any], Any]] = []
    for line in match.group(1).splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
        parts = re.split(r"\s*(?:->|=>|→|==)\s*", line, maxsplit=1)
        if len(parts) != 2:
            continue
        call_text, result_text = parts
        result_text = re.sub(r"\s+#.*$", "", result_text).strip().rstrip(".;")
        try:
            node = ast.parse(call_text, mode="eval").body
            expected = ast.literal_eval(result_text)
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            continue
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        if node.func.id not in names:
            continue
        args = literal_arguments(node)
        if args is not None:
            examples.append((args, expected))
        if len(examples) >= limit:
            break
    return examples


def call_text(name: str, args: list[Any]) -> str:
    return f"{name}({', '.join(repr(a) for a in args)})"


# Finalisation ---------------------------------------------------------------------------------

ACCEPT_EXTRA_ARGS = """

def kloer_accept_extra_args(function):
    import functools
    import inspect

    try:
        parameters = list(inspect.signature(function).parameters.values())
    except (TypeError, ValueError):
        return function
    if any(p.kind == p.VAR_POSITIONAL for p in parameters):
        return function
    limit = sum(p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) for p in parameters)

    @functools.wraps(function)
    def adapted(*args, **kwargs):
        return function(*args[:limit], **kwargs)

    return adapted


{entry_point} = kloer_accept_extra_args({entry_point})
"""

# Kloer 0.4. The function is called directly; only when that raises TypeError is the call
# adapted and tried once more, and if the adapted call fails too, the original error is raised.
# A call that already works therefore behaves exactly as before.
ADAPT_CALLS = r"""

def kloer_adapt_calls(function, fill_missing=True):
    import functools
    import inspect
    import itertools
    import re
    import typing

    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return function
    parameters = list(signature.parameters.values())
    if any(p.kind == p.VAR_POSITIONAL for p in parameters):
        return function
    positional = [p for p in parameters if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    limit = len(positional)
    sized = (list, tuple, str, dict, set, frozenset)
    length_name = re.compile(r"(?i)^(?:n|len|length|size)(?:_\w+)?$|^\w+_(?:n|len|length|size)$")
    simple = {"int": int, "float": float, "str": str, "bool": bool, "list": list,
              "tuple": tuple, "dict": dict, "set": set}

    def kind(annotation):
        if isinstance(annotation, str):
            name = re.split(r"[\[ |]", annotation.strip())[0].split(".")[-1].lower()
            return simple.get(name)
        origin = typing.get_origin(annotation) or annotation
        return origin if origin in simple.values() else None

    kinds = [kind(p.annotation) for p in positional]

    def fits(value, expected):
        if expected in (int, float) and isinstance(value, bool):
            return False
        if expected is float:
            return isinstance(value, (int, float))
        return isinstance(value, expected)

    def binds(args, kwargs):
        try:
            signature.bind(*args, **kwargs)
        except TypeError:
            return False
        return True

    def drop_extra(args):
        # A redundant length (an int equal to the length of another argument) goes first.
        while len(args) > limit:
            for i, value in enumerate(args):
                if type(value) is int and any(
                    j != i and isinstance(other, sized) and len(other) == value
                    for j, other in enumerate(args)
                ):
                    del args[i]
                    break
            else:
                del args[limit:]
        return args

    def fill(args, kwargs):
        lengths = [len(a) for a in args if isinstance(a, sized)]
        for p in positional[len(args):]:
            if p.name in kwargs or p.default is not p.empty:
                break
            args.append(lengths[0] if lengths and length_name.match(p.name) else None)
        return args

    def reorder(args):
        # Only when every parameter is annotated with a simple type and the arguments match
        # those types in another order.
        if not 1 < len(args) == limit <= 4 or None in kinds:
            return args
        if all(map(fits, args, kinds)):
            return args
        for order in itertools.permutations(range(len(args))):
            if all(fits(args[i], expected) for i, expected in zip(order, kinds)):
                return [args[i] for i in order]
        return args

    def adapt(args, kwargs):
        if binds(args, kwargs):
            reordered = reorder(args)
            return None if reordered is args else reordered
        if len(args) > limit:
            args = drop_extra(args)
        elif fill_missing:
            args = fill(args, kwargs)
        args = reorder(args)
        return args if binds(args, kwargs) else None

    @functools.wraps(function)
    def adapted(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except TypeError as exc:
            original = exc
        new_args = adapt(list(args), kwargs)
        if new_args is None:
            raise original
        try:
            return function(*new_args, **kwargs)
        except Exception:
            raise original from None

    return adapted


{entry_point} = kloer_adapt_calls({entry_point}, fill_missing={fill_missing})
"""

# Words that show the task or the user asked for exceptions; input checks are then kept.
RAISE_WORDS = re.compile(r"(?i)\b(?:raise[sd]?|raising|exceptions?|throws?|thrown|errors?)\b")


def is_guard(node: ast.stmt) -> bool:
    """An input check that only raises: `if <test>: raise ...` without else, or an assert."""
    if any(isinstance(n, ast.NamedExpr) for n in ast.walk(node)):
        return False
    if isinstance(node, ast.Assert):
        return True
    return (
        isinstance(node, ast.If)
        and not node.orelse
        and all(isinstance(s, ast.Raise) for s in node.body)
    )


def strip_guards(function: ast.FunctionDef) -> int:
    """Removes the guards at the top level of the function's body; returns how many."""
    kept = [s for s in function.body if not is_guard(s)]
    removed = len(function.body) - len(kept)
    if removed:
        function.body = kept or [ast.Pass()]
    return removed


def finalize(
    code: str,
    problem: dict[str, Any],
    answers: str = "",
    adapt_calls: bool = True,
    fill_missing: bool = True,
    strip_raises: bool = True,
) -> tuple[str, dict[str, Any]]:
    """Deterministic robustness for the evaluation harness; returns the code and what changed.

    - Guards that raise (`if <check>: raise ...`, `assert ...`) at the top level of the entry
      function are removed, unless the task or the user's answers mention exceptions or errors:
      the tests check returned values, so a check nobody asked for can only turn a call into a
      failure (strip_raises).
    - `from collections import Counter`: some MBPP expected outputs are Counter
      objects that the test script builds without importing it.
    - A lower-case alias of the entry point (a prompt says left_Rotate, its tests call
      left_rotate), unless it would shadow a builtin (Sum -> sum) or a keyword.
    - For tasks without a signature, a call that raises TypeError is adapted and tried once more:
      a redundant length argument is dropped (f(arr, n, k) -> f(arr, k)), then extra trailing
      arguments; arguments are reordered to match simple type annotations; with fill_missing, a
      missing length-like parameter gets the length of the first collection argument and other
      missing parameters get None. With adapt_calls=False, extra trailing arguments are only cut
      (Kloer 0.3.1). Only calls that would otherwise fail behave differently.
    """
    notes: dict[str, Any] = {}
    entry_point = problem["entry_point"]
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, notes
    function = entry_function(tree, entry_point)
    if function is None:
        return code, notes
    if strip_raises and not RAISE_WORDS.search(problem["prompt"] + "\n" + answers):
        removed = strip_guards(function)
        if removed:
            notes["guards"] = removed
            code = ast.unparse(tree)
    names = top_level_names(tree)
    parts = [code.rstrip() + "\n"]
    if "Counter" not in names:  # appended, so that a `from __future__` import stays first
        parts.append("\nfrom collections import Counter\n")
    if not has_signature(problem["prompt"], entry_point):
        if adapt_calls:
            adapter = ADAPT_CALLS.replace("{fill_missing}", repr(bool(fill_missing)))
        else:
            adapter = ACCEPT_EXTRA_ARGS
        parts.append(adapter.replace("{entry_point}", entry_point))
    lower = entry_point.lower()
    shadows = lower in dir(builtins) or keyword.iskeyword(lower)  # e.g. Sum -> sum
    if lower != entry_point and lower not in names and lower.isidentifier() and not shadows:
        parts.append(f"\n{lower} = {entry_point}\n")
    return "".join(parts), notes


def finalize_code(code: str, problem: dict[str, Any], answers: str = "", **options: bool) -> str:
    return finalize(code, problem, answers, **options)[0]


# Algorithm ------------------------------------------------------------------------------------


def new_exploration() -> dict[str, Any]:
    return {"candidates": [], "arity": None, "offered": None}


class Kloer(ClarificationAlgorithmBase):
    DEFAULT_CONFIG: dict[str, Any] = {
        "max_questions": 1,  # upper bound on clarification turns (also capped by env.can_ask)
        "always_ask": True,  # ask even if the analyst finds no unresolved point
        "generation_attempts": 3,  # coder retries when no valid python block is returned
        "repair_spec": True,  # implement from a repaired, structured specification
        "neutralize_name": True,  # neutral function name in the spec, when the task has a signature
        "coder_sees_original": False,  # also show the original task to the spec coder
        "use_execution": True,  # explore candidates and verify by execution (False = Kloer 0.2)
        "num_interpretations": 3,  # candidate implementations, one per reading
        "num_probes": 8,  # probe inputs used to compare the candidates
        "exec_timeout_s": 2.0,  # time limit per call inside the execution harness
        "prefer_contradiction": True,  # keep the analyst's question when it is about a contradiction
        "arity_strategy": "optional",  # argument-count disagreement: "optional", "question" or "off"
        "verify_final": True,  # run the final code on the specification's examples
        "self_debug": True,  # one repair round from execution feedback
        "finalize": True,  # harness robustness: Counter import, lower-case alias, call adapter
        # 0.4
        "no_raise_options": True,  # questions never offer "raise an error" unless the task does
        "structure_last": True,  # a string-vs-list-of-its-parts difference is asked about last
        "example_pattern_rule": True,  # rewrite drops examples contradicting the answer by a pattern
        "call_forms": True,  # no signature: infer plausible call forms, the code accepts all
        "adapt_calls": True,  # no signature: drop redundant lengths, reorder by annotations
        "fill_missing_args": True,  # no signature: fill a missing length (or None) instead of failing
        "strip_guard_raises": True,  # remove input checks that raise, unless exceptions were asked for
        "time_budget_s": 240.0,  # skip optional steps when this would be exceeded (0 = no guard)
        "trace": True,  # append a one-line comment recording the decisions taken
        "debug_dir": "",  # development only: write one JSON file per task with every step
    }

    # Estimated model and sandbox calls still needed by each step and everything after it.
    COST_EXPLORE = 10
    COST_CALL_FORMS = 4
    COST_REPAIR = 3
    COST_VERIFY = 3

    def format_clarifications(self, clarifications: list[tuple[str, str]]) -> str:
        if not clarifications:
            return "(none)"
        return "\n".join(f"Q: {q}\nA: {a}" for q, a in clarifications)

    def fill(self, template: str, problem: dict[str, Any], **values: str) -> str:
        """Fills placeholders in order; the task text goes last, so braces inside it are kept."""
        text = template
        for key, value in values.items():  # values may themselves contain later placeholders
            text = text.replace("{" + key + "}", value)
        return text.replace("{entry_point}", problem["entry_point"]).replace(
            "{prompt}", problem["prompt"]
        )

    def no_raise(self, rule: str) -> str:
        return rule if self.config["no_raise_options"] else ""

    # Time guard ---------------------------------------------------------------------------------

    def start_clock(self) -> None:
        self.clock = {"start": time.monotonic(), "calls": 0}

    def tick(self) -> None:
        clock = getattr(self, "clock", None)
        if clock is not None:
            clock["calls"] += 1

    def affordable(self, calls: int) -> bool:
        """False if `calls` more model or sandbox calls, at the average duration so far, would
        exceed the time budget. The runner gives each task a fixed time (300 s by default) and
        restarts tasks that exceed it, so optional steps are skipped rather than risk that."""
        budget = float(self.config["time_budget_s"] or 0)
        clock = getattr(self, "clock", None)
        if budget <= 0 or clock is None:
            return True
        elapsed = time.monotonic() - clock["start"]
        average = elapsed / max(clock["calls"], 1)
        return elapsed + calls * average <= budget

    def llm(self, env: ClarificationEnvironment, prompt: Any) -> str:
        try:
            return env.llm(prompt)
        finally:
            self.tick()

    # 1. Analyse ---------------------------------------------------------------------------------

    def analyse(
        self,
        env: ClarificationEnvironment,
        problem: dict[str, Any],
        clarifications: list[tuple[str, str]],
    ) -> dict[str, Any]:
        prompt = self.fill(
            ANALYST_PROMPT,
            problem,
            no_raise_rule=self.no_raise(NO_RAISE_ANALYST_RULE),
            clarifications=self.format_clarifications(clarifications),
        )
        return parse_json_object(self.llm(env, prompt))

    def fallback_question(self, env: ClarificationEnvironment, problem: dict[str, Any]) -> str:
        prompt = self.fill(
            FALLBACK_QUESTION_PROMPT, problem, no_raise_rule=self.no_raise(NO_RAISE_FALLBACK_RULE)
        )
        return clean_question(self.llm(env, prompt))

    # 2. Explore ---------------------------------------------------------------------------------

    def execute(
        self,
        env: ClarificationEnvironment,
        codes: list[str],
        entry_point: str,
        inputs: list[list[Any]],
        expected: list[Any] | None = None,
    ) -> list[list[list[str]] | None] | None:
        """Runs every code on every input in one sandboxed execution; None if nothing ran."""
        script = build_harness(codes, entry_point, inputs, expected, self.config["exec_timeout_s"])
        try:
            output = env.exec_code(script)
        except Exception:
            return None
        finally:
            self.tick()
        rows = parse_harness_output(str(output), len(codes))
        return rows if any(row is not None for row in rows) else None

    def write_candidates(
        self, env: ClarificationEnvironment, problem: dict[str, Any], readings: list[str]
    ) -> list[dict[str, Any]]:
        entry_point = problem["entry_point"]
        candidates = []
        for reading in readings:
            content = self.fill(CANDIDATE_PROMPT, problem, reading=reading)
            code, _ = self.generate(env, content, entry_point, entry_point, attempts=1)
            if code is not None:
                candidates.append(
                    {"reading": reading, "code": code, "params": entry_params(code, entry_point)}
                )
        return candidates

    def probe_inputs(
        self, env: ClarificationEnvironment, problem: dict[str, Any], candidate: dict[str, Any]
    ) -> list[list[Any]]:
        entry_point = problem["entry_point"]
        params = candidate["params"]
        if params is None:
            return []
        required = required_count(candidate["code"], entry_point)
        prompt = self.fill(
            PROBE_PROMPT,
            problem,
            n=str(self.config["num_probes"]),
            signature=signature_line(candidate["code"], entry_point),
        )
        return parse_probe_inputs(
            self.llm(env, prompt),
            entry_point,
            len(params) if required is None else required,
            len(params),
            self.config["num_probes"],
        )

    def value_question(
        self,
        env: ClarificationEnvironment,
        problem: dict[str, Any],
        offered: dict[str, str],
        reading_a: str,
        reading_b: str,
    ) -> str:
        values = {"call": offered["call"], "value_a": offered["a"], "value_b": offered["b"]}
        prompt = self.fill(
            VALUE_QUESTION_PROMPT, problem, reading_a=reading_a, reading_b=reading_b, **values
        )
        question = clean_question(self.llm(env, prompt))
        if not question.endswith("?") or len(question) > 500:
            template = VALUE_QUESTION_TEMPLATE
            for key, value in values.items():
                template = template.replace("{" + key + "}", value)
            question = clean_question(template)
        return question

    def explore(
        self,
        env: ClarificationEnvironment,
        problem: dict[str, Any],
        analysis: dict[str, Any],
        exploration: dict[str, Any],
        trace: dict[str, Any],
        ask: bool = True,
    ) -> str | None:
        """Writes and runs candidates; returns an execution-grounded question, or None.

        With ask=False the candidates are still written (for the arity rule and for verification)
        but no question is derived from them.
        """
        entry_point = problem["entry_point"]
        readings = analysis.get("interpretations")
        readings = (
            [str(r).strip() for r in readings if str(r).strip()]
            if isinstance(readings, list)
            else []
        )
        readings = list(dict.fromkeys(readings))[: self.config["num_interpretations"]]
        if len(readings) < 2:
            trace["explore"] = "no-readings"
            return None
        candidates = self.write_candidates(env, problem, readings)
        exploration["candidates"] = candidates
        trace["cands"] = len(candidates)
        if len(candidates) < 2:
            return None

        # Argument count (tasks without a signature only).
        known = [c["params"] for c in candidates if c["params"] is not None]
        strategy = self.config["arity_strategy"]
        if (
            strategy != "off"
            and not has_signature(problem["prompt"], entry_point)
            and len({len(p) for p in known}) > 1
        ):
            short, long = min(known, key=len), max(known, key=len)
            exploration["arity"] = (short, long)
            trace["arity"] = [len(short), len(long)]
            if strategy == "question" and ask:
                return clean_question(
                    ARITY_QUESTION_TEMPLATE.replace("{entry_point}", entry_point)
                    .replace("{short}", ", ".join(short))
                    .replace("{long}", ", ".join(long))
                )

        if not ask:
            return None

        # Behaviour on probe inputs.
        inputs = self.probe_inputs(env, problem, candidates[0])
        trace["probes"] = len(inputs)
        if not inputs:
            return None
        rows = self.execute(env, [c["code"] for c in candidates], entry_point, inputs)
        exploration["probe_inputs"] = [repr(i) for i in inputs]
        exploration["probe_rows"] = rows
        if rows is None:
            trace["exec"] = "failed"
            return None
        found = distinguishing_input(rows, structure_last=self.config["structure_last"])
        if found is None:
            trace["groups"] = 1
            return None
        position, a, b, groups = found
        trace["groups"] = groups
        offered = {
            "args": inputs[position],
            "call": call_text(entry_point, inputs[position]),
            "a": rows[a][position][1],
            "b": rows[b][position][1],
        }
        exploration["offered"] = offered
        return self.value_question(
            env, problem, offered, candidates[a]["reading"], candidates[b]["reading"]
        )

    # 3. Ask -------------------------------------------------------------------------------------

    def clarify(
        self, env: ClarificationEnvironment, problem: dict[str, Any], trace: dict[str, Any]
    ) -> tuple[list[tuple[str, str]], list[str], dict[str, Any]]:
        clarifications: list[tuple[str, str]] = []
        assumptions: list[str] = []
        exploration = new_exploration()
        for _ in range(self.config["max_questions"]):
            if not env.can_ask():
                break
            analysis = self.analyse(env, problem, clarifications)
            exploration["analysis"] = analysis
            points = analysis.get("points", analysis.get("issues", []))
            points = [p for p in points if isinstance(p, dict)] if isinstance(points, list) else []
            assumptions = [
                str(p["assumption"]) for p in points if p.get("resolvable") and p.get("assumption")
            ]
            question = clean_question(str(analysis.get("question") or ""))
            source = "analyst"
            # A quoted contradiction between two statements is asked about directly.
            contradiction = any(
                "contradict" in str(p.get("type", "")).lower()
                and not p.get("resolvable", False)
                and str(p.get("source_b") or "").strip()
                for p in points
            )
            keep_analyst = bool(question) and contradiction and self.config["prefer_contradiction"]
            if keep_analyst:
                source = "contradiction"
            if self.config["use_execution"] and not clarifications:
                if self.affordable(self.COST_EXPLORE):
                    explored = self.explore(
                        env, problem, analysis, exploration, trace, ask=not keep_analyst
                    )
                    if explored:
                        question = explored
                        source = "arity" if exploration["offered"] is None else "value"
                else:
                    trace.setdefault("time", []).append("no-explore")
            has_open_point = any(not p.get("resolvable", False) for p in points)
            if not (has_open_point or self.config["always_ask"]):
                break
            if not question and not clarifications:
                # Analysis failed or returned no question: still use the turn.
                question = self.fallback_question(env, problem) or self.fallback_question(
                    env, problem
                )
                source = "fallback"
                if not question:
                    question = GENERIC_QUESTION.replace("{entry_point}", problem["entry_point"])
                    source = "generic"
            if not question:
                break
            try:
                answer = env.ask_human(question)
            except TooManyQuestionException:
                break
            finally:
                self.tick()
            trace["q"] = source
            clarifications.append((question, answer))
        return clarifications, assumptions, exploration

    # 4. Repair ----------------------------------------------------------------------------------

    def extra_rules(self, exploration: dict[str, Any], for_repair: bool) -> str:
        rules = []
        arity = exploration.get("arity")
        if arity and self.config["arity_strategy"] == "optional":
            short, long = arity
            rules.append(
                ARITY_RULE.replace("{k}", str(len(short)))
                .replace("{m}", str(len(long)))
                .replace("{params}", ", ".join(long))
            )
        forms = exploration.get("forms")
        if forms:
            rules.append(
                CALL_FORMS_RULE.replace("{forms}", "; ".join(forms)).replace(
                    "{entry_point}", forms[0].split("(", 1)[0]
                )
            )
        offered = exploration.get("offered")
        if offered and for_repair:
            rules.append(
                OFFERED_RULE.replace("{call}", offered["call"])
                .replace("{value_a}", offered["a"])
                .replace("{value_b}", offered["b"])
            )
        if for_repair and self.config["example_pattern_rule"]:
            rules.append(EXAMPLE_PATTERN_RULE)
        return "".join("\n- " + rule for rule in rules)

    def call_forms(
        self,
        env: ClarificationEnvironment,
        problem: dict[str, Any],
        clarifications: list[tuple[str, str]],
    ) -> list[str]:
        """Plausible call forms of a task without a signature (empty if only one is plausible)."""
        prompt = self.fill(
            CALL_FORMS_PROMPT, problem, clarifications=self.format_clarifications(clarifications)
        )
        forms = parse_call_forms(self.llm(env, prompt), problem["entry_point"])
        return forms if len(forms) > 1 else []

    def repair(
        self,
        env: ClarificationEnvironment,
        problem: dict[str, Any],
        clarifications: list[tuple[str, str]],
        assumptions: list[str],
        exploration: dict[str, Any],
    ) -> tuple[str | None, str]:
        """Returns the repaired specification (None on failure) and the function name it uses."""
        entry_point = problem["entry_point"]
        neutral = self.config["neutralize_name"] and has_signature(problem["prompt"], entry_point)
        name = NEUTRAL_NAME if neutral and entry_point != NEUTRAL_NAME else entry_point
        naming_rule = NAMING_RULE_NEUTRAL if name != entry_point else NAMING_RULE_KEEP
        prompt = self.fill(
            REPAIR_PROMPT,
            problem,
            naming_rule=naming_rule,
            extra_rules=self.extra_rules(exploration, for_repair=True),
            name=name,
            clarifications=self.format_clarifications(clarifications),
            assumptions="\n".join(f"- {a}" for a in assumptions) or "(none)",
        )
        return clean_spec(self.llm(env, prompt), entry_point, name), name

    # 5. Implement and verify --------------------------------------------------------------------

    def generate(
        self,
        env: ClarificationEnvironment,
        content: str,
        entry_point: str,
        name: str,
        attempts: int | None = None,
    ) -> tuple[str | None, str]:
        """Returns the sanitised code (None if no valid implementation) and the last response."""
        messages = [{"role": "user", "content": content}]
        response = ""
        for _ in range(attempts or self.config["generation_attempts"]):
            response = self.llm(env, messages)
            code = extract_code(response, entry_point, name if name != entry_point else None)
            if code is not None:
                return code, response
            messages += [
                {"role": "assistant", "content": response},
                {"role": "user", "content": FORMAT_REMINDER.replace("{name}", name)},
            ]
        return None, response

    def implement_from_spec(
        self,
        env: ClarificationEnvironment,
        problem: dict[str, Any],
        spec: str,
        name: str,
        clarifications: list[tuple[str, str]],
        exploration: dict[str, Any],
    ) -> tuple[str | None, str]:
        original = ""
        if self.config["coder_sees_original"]:
            original = ORIGINAL_SECTION.replace("{prompt}", problem["prompt"])
        content = (  # the task text (inside `original`) goes last, so braces inside it are kept
            SPEC_CODER_PROMPT.replace("{name}", name)
            .replace("{extra_rules}", self.extra_rules(exploration, for_repair=False))
            .replace("{clarifications}", self.format_clarifications(clarifications))
            .replace("{spec}", spec)
            .replace("{original}", original)
        )
        return self.generate(env, content, problem["entry_point"], name)

    def implement(
        self,
        env: ClarificationEnvironment,
        problem: dict[str, Any],
        clarifications: list[tuple[str, str]],
        assumptions: list[str],
    ) -> tuple[str | None, str]:
        """Kloer 0.1 coder, working from the original task (fallback)."""
        entry_point = problem["entry_point"]
        content = self.fill(
            CODER_PROMPT,
            problem,
            clarifications=self.format_clarifications(clarifications),
            assumptions="\n".join(f"- {a}" for a in assumptions) or "(none)",
        )
        return self.generate(env, content, entry_point, entry_point)

    def debug(
        self,
        env: ClarificationEnvironment,
        problem: dict[str, Any],
        spec: str,
        name: str,
        clarifications: list[tuple[str, str]],
        code: str,
        checks: list[tuple[list[Any], Any]],
        row: list[list[str]],
    ) -> str | None:
        failures = []
        for (args, expected), cell in zip(checks, row, strict=False):
            if cell[0] != "pass":
                got = cell[1] if cell[0] == "fail" else f"raised {cell[1]}"
                failures.append(f"- {call_text(name, args)}: expected {expected!r}, got {got}")
        if not failures:
            return None
        content = (
            DEBUG_PROMPT.replace("{name}", name)
            .replace("{failures}", "\n".join(failures))
            .replace("{clarifications}", self.format_clarifications(clarifications))
            .replace("{spec}", spec)
            .replace("{code}", code)
        )
        fixed, _ = self.generate(env, content, problem["entry_point"], name, attempts=1)
        return fixed

    def verify(
        self,
        env: ClarificationEnvironment,
        problem: dict[str, Any],
        spec: str,
        name: str,
        clarifications: list[tuple[str, str]],
        code: str,
        exploration: dict[str, Any],
        trace: dict[str, Any],
    ) -> str:
        """Checks the final code on the result the user confirmed for the offered call.

        Only that example is trusted: the specification's other examples come from the task
        (deliberately wrong in contradictory variants) or from the model. If the code fails it,
        one self-debugging round is tried and kept only if it passes.
        """
        offered = exploration.get("offered")
        if not offered:
            return code
        entry_point = problem["entry_point"]
        checks = [
            check
            for check in parse_spec_examples(spec, {name, entry_point})
            if repr(check[0]) == repr(offered["args"])
        ][:1]
        trace["checks"] = len(checks)
        if not checks:
            return code
        inputs, expected = [checks[0][0]], [checks[0][1]]
        rows = self.execute(env, [code], entry_point, inputs, expected)
        exploration["verify_rows"] = rows
        if rows is None or rows[0] is None:
            trace["verify"] = "exec-failed"
            return code
        if rows[0][0][0] == "pass":
            trace["verify"] = "final-pass"
            return code
        if self.config["self_debug"]:
            fixed = self.debug(env, problem, spec, name, clarifications, code, checks, rows[0])
            exploration["debug_code"] = fixed
            if fixed is not None:
                again = self.execute(env, [fixed], entry_point, inputs, expected)
                if again and again[0] and again[0][0][0] == "pass":
                    trace["verify"] = "debug-pass"
                    return fixed
        trace["verify"] = "kept-failing"
        return code

    def run(self, env: ClarificationEnvironment, problem: dict[str, Any]) -> str:
        trace: dict[str, Any] = {"v": KLOER_VERSION, "q": "none"}
        clarifications: list[tuple[str, str]] = []
        assumptions: list[str] = []
        exploration = new_exploration()
        code: str | None = None
        response = ""
        spec: str | None = None
        self.start_clock()
        try:
            clarifications, assumptions, exploration = self.clarify(env, problem, trace)
            name = problem["entry_point"]
            if self.config["call_forms"] and not has_signature(problem["prompt"], name):
                if self.affordable(self.COST_CALL_FORMS):
                    exploration["forms"] = self.call_forms(env, problem, clarifications)
                    trace["forms"] = len(exploration["forms"])
                else:
                    trace.setdefault("time", []).append("no-forms")
            if self.config["repair_spec"]:
                if self.affordable(self.COST_REPAIR):
                    spec, name = self.repair(env, problem, clarifications, assumptions, exploration)
                    if spec is not None:
                        code, response = self.implement_from_spec(
                            env, problem, spec, name, clarifications, exploration
                        )
                else:
                    trace.setdefault("time", []).append("no-repair")
            if code is None:
                trace["coder"] = "fallback"
                code, response = self.implement(env, problem, clarifications, assumptions)
            if (
                code is not None
                and spec is not None
                and self.config["use_execution"]
                and self.config["verify_final"]
            ):
                if self.affordable(self.COST_VERIFY):
                    code = self.verify(
                        env, problem, spec, name, clarifications, code, exploration, trace
                    )
                elif exploration.get("offered"):
                    trace.setdefault("time", []).append("no-verify")
        except LimitsExceededException:
            trace["limit"] = True
        result = response if code is None else code
        if code is not None and self.config["finalize"]:
            result, notes = finalize(
                code,
                problem,
                "\n".join(answer for _, answer in clarifications),
                adapt_calls=self.config["adapt_calls"],
                fill_missing=self.config["fill_missing_args"],
                strip_raises=self.config["strip_guard_raises"],
            )
            trace.update(notes)
        if code is not None and self.config["trace"]:
            result += "\n\n# kloer-trace: " + json.dumps(trace)
        if self.config["debug_dir"]:
            self.write_debug_record(
                problem, trace, clarifications, assumptions, exploration, spec, code, result
            )
        return result

    def write_debug_record(
        self,
        problem: dict[str, Any],
        trace: dict[str, Any],
        clarifications: list[tuple[str, str]],
        assumptions: list[str],
        exploration: dict[str, Any],
        spec: str | None,
        code: str | None,
        result: str,
    ) -> None:
        """Development only: one JSON file per task (named by a hash of the task) with every
        intermediate step. Never raises."""
        try:
            key = problem["entry_point"] + "\n" + problem["prompt"].strip()
            digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
            record = {
                "entry_point": problem["entry_point"],
                "prompt": problem["prompt"],
                "trace": trace,
                "clarifications": clarifications,
                "assumptions": assumptions,
                "exploration": exploration,
                "spec": spec,
                "code_before_finalize": code,
                "result": result,
            }
            os.makedirs(self.config["debug_dir"], exist_ok=True)
            path = os.path.join(self.config["debug_dir"], digest + ".json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(record, f, default=repr, indent=1)
        except Exception:
            pass
