# SPDX-FileCopyrightText: 2026 Guillaume Haben <guillaume.haben@telecomnancy.net>
#
# SPDX-License-Identifier: MIT

"""Kloer.

Analyse, ask, repair, implement.

1. Analyse. A requirements-analyst call, decoupled from code generation, reviews the task
   against an ambiguity checklist (contradictions plus the ambiguity taxonomy of
   ClarifyCodeBench, Fang et al. 2026), resolves what the examples, signature or convention
   already settle, and picks the one open point most likely to break hidden tests.
2. Ask. One atomic question about that point. When the point is a contradiction, the question
   quotes the two conflicting statements and asks which one holds (the "which source is
   correct" confirmation of VClare's spec-level repair, Zhao et al. 2026); otherwise it offers
   two concrete alternatives.
3. Repair. The task is rewritten into a structured specification (signature, description,
   inputs, output, rules, examples), editing only what the answer invalidates (VClare's
   targeted repair) and laid out as redundant layers, which makes code generation more robust
   (Akli et al. 2026, "When prompt under-specification improves code correctness"). The rewrite
   applies meaning-preserving rules in IF-THEN-EXCEPT form: terminology clarification, math
   symbols in plain English, binary-predicate completion and function-name neutralisation
   (the universal rules of DualFix, Akli et al. 2026, "From Failing to Passing").
4. Implement. The coder works from the repaired specification; the code is sanitised
   deterministically (fence variants, demo code) and the neutral name is bound back to the
   required entry point.

Team: Kloer
Team Members: Milos Ojdanic, Guillaume Haben, Sami Lazreg
Main Contact: guillaume.haben@telecomnancy.net
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any

from clarify.baselines import ClarificationAlgorithmBase
from clarify.env import (
    ClarificationEnvironment,
    LimitsExceededException,
    TooManyQuestionException,
)
from clarify.runtime import _validate_and_parse_evalplus_result

KLOER_VERSION = "0.2"

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
     backticks or code formatting.
Keep the analysis short.

Answer with a JSON object only:
{"points": [{"type": "...", "source_a": "...", "source_b": "", "resolvable": false, "assumption": "..."}], "question": "..."}
""".strip()

FALLBACK_QUESTION_PROMPT = """
A user asked for a Python function named {entry_point}:

{prompt}

Write the single most important clarifying question whose answer would change what
{entry_point} returns. Ask for exactly one fact, offer the two most plausible concrete
alternatives, and do not ask about implementation details or test cases.
Reply with the question only, as plain text without backticks.
""".strip()

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
  compute new expected results.
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
- {naming_rule}

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
- Do not add input validation, assertions or exceptions that the specification does not require.
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


# Algorithm ------------------------------------------------------------------------------------


class Kloer(ClarificationAlgorithmBase):
    DEFAULT_CONFIG: dict[str, Any] = {
        "max_questions": 1,  # upper bound on clarification turns (also capped by env.can_ask)
        "always_ask": True,  # ask even if the analyst finds no unresolved point
        "generation_attempts": 3,  # coder retries when no valid python block is returned
        "repair_spec": True,  # implement from a repaired, structured specification
        "neutralize_name": True,  # neutral function name in the spec, when the task has a signature
        "coder_sees_original": False,  # also show the original task to the spec coder
    }

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

    # 1-2. Analyse and ask ----------------------------------------------------------------------

    def analyse(
        self,
        env: ClarificationEnvironment,
        problem: dict[str, Any],
        clarifications: list[tuple[str, str]],
    ) -> dict[str, Any]:
        prompt = self.fill(
            ANALYST_PROMPT, problem, clarifications=self.format_clarifications(clarifications)
        )
        return parse_json_object(env.llm(prompt))

    def fallback_question(self, env: ClarificationEnvironment, problem: dict[str, Any]) -> str:
        return clean_question(env.llm(self.fill(FALLBACK_QUESTION_PROMPT, problem)))

    def clarify(
        self, env: ClarificationEnvironment, problem: dict[str, Any]
    ) -> tuple[list[tuple[str, str]], list[str]]:
        clarifications: list[tuple[str, str]] = []
        assumptions: list[str] = []
        for _ in range(self.config["max_questions"]):
            if not env.can_ask():
                break
            analysis = self.analyse(env, problem, clarifications)
            points = analysis.get("points", analysis.get("issues", []))
            points = [p for p in points if isinstance(p, dict)] if isinstance(points, list) else []
            assumptions = [
                str(p["assumption"]) for p in points if p.get("resolvable") and p.get("assumption")
            ]
            question = clean_question(str(analysis.get("question") or ""))
            has_open_point = any(not p.get("resolvable", False) for p in points)
            if not (has_open_point or self.config["always_ask"]):
                break
            if not question and not clarifications:
                # Analysis failed or returned no question: still use the turn.
                question = self.fallback_question(env, problem)
            if not question:
                break
            try:
                answer = env.ask_human(question)
            except TooManyQuestionException:
                break
            clarifications.append((question, answer))
        return clarifications, assumptions

    # 3. Repair ----------------------------------------------------------------------------------

    def repair(
        self,
        env: ClarificationEnvironment,
        problem: dict[str, Any],
        clarifications: list[tuple[str, str]],
        assumptions: list[str],
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
            name=name,
            clarifications=self.format_clarifications(clarifications),
            assumptions="\n".join(f"- {a}" for a in assumptions) or "(none)",
        )
        return clean_spec(env.llm(prompt), entry_point, name), name

    # 4. Implement -------------------------------------------------------------------------------

    def generate(
        self, env: ClarificationEnvironment, content: str, entry_point: str, name: str
    ) -> tuple[str | None, str]:
        """Returns the sanitised code (None if no valid implementation) and the last response."""
        messages = [{"role": "user", "content": content}]
        response = ""
        for _ in range(self.config["generation_attempts"]):
            response = env.llm(messages)
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
    ) -> tuple[str | None, str]:
        original = ""
        if self.config["coder_sees_original"]:
            original = ORIGINAL_SECTION.replace("{prompt}", problem["prompt"])
        content = (  # the task text (inside `original`) goes last, so braces inside it are kept
            SPEC_CODER_PROMPT.replace("{name}", name)
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

    def run(self, env: ClarificationEnvironment, problem: dict[str, Any]) -> str:
        clarifications: list[tuple[str, str]] = []
        assumptions: list[str] = []
        code: str | None = None
        response = ""
        try:
            clarifications, assumptions = self.clarify(env, problem)
            if self.config["repair_spec"]:
                spec, name = self.repair(env, problem, clarifications, assumptions)
                if spec is not None:
                    code, response = self.implement_from_spec(
                        env, problem, spec, name, clarifications
                    )
            if code is None:
                code, response = self.implement(env, problem, clarifications, assumptions)
        except LimitsExceededException:
            pass
        return code if code is not None else response
