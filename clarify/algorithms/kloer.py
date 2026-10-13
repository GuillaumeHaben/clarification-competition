# SPDX-FileCopyrightText: 2026 Guillaume Haben <guillaume.haben@telecomnancy.net>
#
# SPDX-License-Identifier: MIT

"""Kloer.

Analyst-then-coder clarification. A first LLM call acts as a requirements
analyst, decoupled from code generation: it lists the contradictory, ambiguous
or incomplete points of the task, discards those that can be inferred from the
examples, signature or convention, and asks the user one atomic question about
the remaining point most likely to change the function's behaviour. A second
LLM call implements the function from the task and the clarification, using a
fixed precedence order to resolve conflicts (clarification > examples >
signature > conventional reading). The output is checked for a fenced Python
block defining the entry point and regenerated if needed.

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

ANALYST_PROMPT = """
You are a requirements analyst reviewing a coding task before any code is written.
The task asks for a Python function named `{entry_point}`.

### Task
{prompt}

### Previous clarifications
{clarifications}

### Instructions
1. List every point where the task is contradictory, ambiguous or incomplete, such that two
   reasonable developers could write versions of `{entry_point}` that return different outputs
   for the same input. Typical points: statements that conflict with each other or with the
   examples, the exact return value or type, the input format or argument order, inclusive vs.
   exclusive bounds, ordering, tie-breaking, duplicates, empty or edge-case inputs, and what to
   return when nothing matches. Ignore style, performance and error handling unless they change
   the returned value.
2. For each point, decide whether it can be resolved confidently from the examples, the function
   name and signature, the previous clarifications, or a strong common convention.
3. Pick the ONE unresolved point that is most likely to make hidden tests fail if guessed wrong.
   Explicit contradictions matter more than vague wording.
4. Write one clarifying question about that point. The question must:
   - ask for exactly one fact (no lists, no "and also", no sub-questions);
   - ask about the required behaviour, never about the implementation, algorithm or test cases;
   - be self-contained and answerable in one sentence, ideally offering the two most plausible
     options (e.g. "Should X return A or B when ...?").

Answer with a JSON object only:
{"issues": [{"point": "...", "resolvable": true, "assumption": "..."}], "question": "..."}
""".strip()

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
    "Your answer must contain the full implementation of `{entry_point}` in a single "
    "```python block. Please answer again."
)


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


def extract_code(response: str, entry_point: str) -> str | None:
    """Returns the code defining `entry_point`, tolerating fence variants such as
    "``` python", "```py" or an untagged ``` block (common with local models)."""
    normalized = re.sub(r"```[ \t]*(?:python3?|py)\b", "```python", response, flags=re.IGNORECASE)
    try:
        code = _validate_and_parse_evalplus_result(normalized)
        if defines_entry_point(code, entry_point):
            return code
    except ValueError:
        pass
    for block in re.findall(r"```[^\n]*\n(.*?)```", normalized, flags=re.DOTALL):
        if defines_entry_point(block, entry_point):
            return block
    if defines_entry_point(response, entry_point):  # bare code without any fence
        return response
    return None


def defines_entry_point(code: str, entry_point: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry_point
        for node in ast.walk(tree)
    )


class Kloer(ClarificationAlgorithmBase):
    DEFAULT_CONFIG: dict[str, Any] = {
        "max_questions": 1,  # upper bound on clarification turns (also capped by env.can_ask)
        "always_ask": True,  # ask even if the analyst finds no unresolved point
        "generation_attempts": 3,  # coder retries when no valid python block is returned
    }

    def format_clarifications(self, clarifications: list[tuple[str, str]]) -> str:
        if not clarifications:
            return "(none)"
        return "\n".join(f"Q: {q}\nA: {a}" for q, a in clarifications)

    def analyse(
        self,
        env: ClarificationEnvironment,
        problem: dict[str, Any],
        clarifications: list[tuple[str, str]],
    ) -> dict[str, Any]:
        prompt = (
            ANALYST_PROMPT.replace("{entry_point}", problem["entry_point"])
            .replace("{prompt}", problem["prompt"])
            .replace("{clarifications}", self.format_clarifications(clarifications))
        )
        return parse_json_object(env.llm(prompt))

    def implement(
        self,
        env: ClarificationEnvironment,
        problem: dict[str, Any],
        clarifications: list[tuple[str, str]],
        assumptions: list[str],
    ) -> str:
        entry_point = problem["entry_point"]
        content = (
            CODER_PROMPT.replace("{entry_point}", entry_point)
            .replace("{prompt}", problem["prompt"])
            .replace("{clarifications}", self.format_clarifications(clarifications))
            .replace("{assumptions}", "\n".join(f"- {a}" for a in assumptions) or "(none)")
        )
        messages = [{"role": "user", "content": content}]
        response = ""
        for _ in range(self.config["generation_attempts"]):
            response = env.llm(messages)
            code = extract_code(response, entry_point)
            if code is not None:
                return code
            messages += [
                {"role": "assistant", "content": response},
                {"role": "user", "content": FORMAT_REMINDER.replace("{entry_point}", entry_point)},
            ]
        return response

    def run(self, env: ClarificationEnvironment, problem: dict[str, Any]) -> str:
        clarifications: list[tuple[str, str]] = []
        assumptions: list[str] = []

        try:
            for _ in range(self.config["max_questions"]):
                if not env.can_ask():
                    break
                analysis = self.analyse(env, problem, clarifications)
                issues = [i for i in analysis.get("issues", []) if isinstance(i, dict)]
                assumptions = [
                    str(i["assumption"])
                    for i in issues
                    if i.get("resolvable") and i.get("assumption")
                ]
                question = str(analysis.get("question") or "").strip()
                has_open_issue = any(not i.get("resolvable", False) for i in issues)
                if not question or not (has_open_issue or self.config["always_ask"]):
                    break
                try:
                    answer = env.ask_human(question)
                except TooManyQuestionException:
                    break
                clarifications.append((question, answer))
        except LimitsExceededException:
            pass

        return self.implement(env, problem, clarifications, assumptions)
