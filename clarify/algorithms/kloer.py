# SPDX-FileCopyrightText: 2026 Guillaume Haben <guillaume.haben@telecomnancy.net>
#
# SPDX-License-Identifier: MIT

"""Kloer.

Analyst-then-coder clarification. A first LLM call acts as a requirements
analyst, decoupled from code generation: it lists the contradictory, ambiguous
or incomplete points of the task, discards those that can be inferred from the
examples, signature or convention, and asks the user one atomic question about
the remaining point most likely to change the function's behaviour, phrased as
two concrete alternatives. A second LLM call implements the function from the
task and the clarification, using a fixed precedence order to resolve conflicts
(clarification > examples > signature > conventional reading). The returned
code is sanitised deterministically (fence variants, demo code, renamed entry
point) and regenerated if no valid implementation is found.

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

KLOER_VERSION = "0.1"

ANALYST_PROMPT = """
You are a requirements analyst reviewing a coding task before any code is written.
The task asks for a Python function named {entry_point}.

### Task
{prompt}

### Previous clarifications
{clarifications}

### Instructions
1. List every point where the task is contradictory, ambiguous or incomplete, such that two
   reasonable developers could write versions of {entry_point} that return different outputs
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
   - ask for exactly one fact (no lists, no "and also", no sub-questions, a single question mark);
   - ask about the required behaviour, never about the implementation, algorithm or test cases;
   - offer the two most plausible concrete alternatives (e.g. "Should X return A or B when ...?");
   - never be a yes/no question that only confirms the obvious reading of the task;
   - be self-contained, answerable in one sentence, and written as plain text without backticks
     or code formatting.

Answer with a JSON object only:
{"issues": [{"point": "...", "resolvable": true, "assumption": "..."}], "question": "..."}
""".strip()

FALLBACK_QUESTION_PROMPT = """
A user asked for a Python function named {entry_point}:

{prompt}

Write the single most important clarifying question whose answer would change what
{entry_point} returns. Ask for exactly one fact, offer the two most plausible concrete
alternatives, and do not ask about implementation details or test cases.
Reply with the question only, as plain text without backticks.
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


def sanitize_code(code: str, entry_point: str) -> str | None:
    """Removes demo code and binds the entry point; returns None if it cannot be bound."""
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
        target = same_name[0] if same_name else (functions[0] if len(functions) == 1 else None)
        if target is None:
            return None
        tree.body.append(ast.parse(f"{entry_point} = {target}").body[0])
        changed = True

    return ast.unparse(tree) if changed else code


def extract_code(response: str, entry_point: str) -> str | None:
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
        code = sanitize_code(block, entry_point)
        if code is not None:
            return code
    return None


# Algorithm ------------------------------------------------------------------------------------


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

    def fill(self, template: str, problem: dict[str, Any], **values: str) -> str:
        text = template.replace("{entry_point}", problem["entry_point"]).replace(
            "{prompt}", problem["prompt"]
        )
        for key, value in values.items():
            text = text.replace("{" + key + "}", value)
        return text

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

    def implement(
        self,
        env: ClarificationEnvironment,
        problem: dict[str, Any],
        clarifications: list[tuple[str, str]],
        assumptions: list[str],
    ) -> str:
        entry_point = problem["entry_point"]
        content = self.fill(
            CODER_PROMPT,
            problem,
            clarifications=self.format_clarifications(clarifications),
            assumptions="\n".join(f"- {a}" for a in assumptions) or "(none)",
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
                question = clean_question(str(analysis.get("question") or ""))
                has_open_issue = any(not i.get("resolvable", False) for i in issues)
                if not (has_open_issue or self.config["always_ask"]):
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
        except LimitsExceededException:
            pass

        return self.implement(env, problem, clarifications, assumptions)
