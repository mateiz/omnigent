"""Tiny ARC-AGI evaluator for Otto's GEPA smoke harness."""

from __future__ import annotations

import importlib.util
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import litellm
from databricks.sdk.config import Config
from datasets import load_dataset
from litellm import completion

MODEL = "databricks/databricks-gemini-3-flash"
PROFILE = "oss"
BACKGROUND = """You are optimizing a small ARC-AGI solving agent program.

The candidate artifact is Python source defining:

    solve(train_inputs, train_outputs, test_inputs, llm) -> dict

It receives ARC training grids, hidden test inputs, and an llm(prompt) callable.
It should return {"train": [...], "test": [[attempt1, attempt2], ...]}.
A task is solved only when every test output has a correct attempt.
"""
OBJECTIVE = "Maximize the fraction of ARC-AGI validation tasks solved."


@dataclass
class ArcExample:
    problem_id: str
    train_in: list[Any]
    train_out: list[Any]
    test_in: list[Any]
    test_out: list[Any]


@dataclass
class TrackedLLM:
    model: str = MODEL
    max_calls: int = 4
    profile: str = PROFILE
    calls: list[dict[str, Any]] = field(default_factory=list)

    def __call__(self, prompt: str, temperature: float = 0.2) -> str:
        if len(self.calls) >= self.max_calls:
            raise RuntimeError(f"LLM call budget exhausted: {self.max_calls}")
        start = time.time()
        host, token = databricks_credentials(self.profile)
        completion_kwargs: dict[str, Any] = {}
        if host and token:
            completion_kwargs.update({"api_base": host, "api_key": token})
        response = completion(
            model=self.model,
            custom_llm_provider="databricks",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            **completion_kwargs,
        )
        content = response.choices[0].message.content or ""
        try:
            cost = litellm.completion_cost(completion_response=response)
        except Exception:  # noqa: BLE001 - cost metadata is best-effort only.
            cost = 0.0
        self.calls.append(
            {"prompt": prompt, "response": content, "cost": cost, "duration": time.time() - start}
        )
        return content

    @property
    def total_cost(self) -> float:
        return sum(float(call.get("cost", 0.0)) for call in self.calls)


def load_arc_slices(
    train_size: int,
    val_size: int,
    seed: int,
) -> tuple[list[ArcExample], list[ArcExample]]:
    dataset = load_dataset("dataartist/arc-agi")

    def convert(raw: dict[str, Any]) -> ArcExample:
        return ArcExample(
            problem_id=raw["id"],
            train_in=[item["input"] for item in raw["train"]],
            train_out=[item["output"] for item in raw["train"]],
            test_in=[item["input"] for item in raw["test"]],
            test_out=[item["output"] for item in raw["test"]],
        )

    examples = [convert(item) for item in dataset["training"].shuffle(seed=seed)]
    return examples[:train_size], examples[train_size : train_size + val_size]


def databricks_credentials(profile: str) -> tuple[str | None, str | None]:
    if os.environ.get("DATABRICKS_HOST") or os.environ.get("DATABRICKS_TOKEN"):
        return os.environ.get("DATABRICKS_HOST"), os.environ.get("DATABRICKS_TOKEN")
    config = Config(profile=profile)
    return config.host, config.token


def ensure_databricks_env(profile: str) -> None:
    os.environ.setdefault("DATABRICKS_CONFIG_PROFILE", profile)
    host, token = databricks_credentials(profile)
    if host and token:
        os.environ.setdefault("DATABRICKS_HOST", host)
        os.environ.setdefault("DATABRICKS_TOKEN", token)


def load_seed(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_baseline_solver(path: Path):
    spec = importlib.util.spec_from_file_location("otto_seed_agent", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.solve


def score_grids(predictions: list[Any], gold: list[Any]) -> float:
    if not gold:
        return 0.0
    correct = sum(
        1
        for prediction, expected in zip(predictions, gold, strict=False)
        if prediction == expected
    )
    return correct / len(gold)


def score_test_attempts(predictions: list[Any], gold: list[Any]) -> float:
    if len(predictions) < len(gold):
        return 0.0
    for attempts, expected in zip(predictions, gold, strict=False):
        normalized = attempts if isinstance(attempts, list) else [attempts]
        if not any(attempt == expected for attempt in normalized[:2]):
            return 0.0
    return 1.0


def run_candidate(
    candidate: str,
    example: ArcExample,
    *,
    model: str,
    max_llm_calls: int,
    profile: str = PROFILE,
) -> dict[str, Any]:
    llm = TrackedLLM(model=model, max_calls=max_llm_calls, profile=profile)
    try:
        namespace: dict[str, Any] = {}
        exec(candidate, namespace)
        result = namespace["solve"](example.train_in, example.train_out, example.test_in, llm)
        train_predictions = result.get("train", [])
        test_predictions = result.get("test", [])
        train_score = score_grids(train_predictions, example.train_out)
        test_score = score_test_attempts(test_predictions, example.test_out)
        error = None
    except Exception as exc:  # noqa: BLE001 - candidate artifacts are arbitrary Python.
        train_predictions = []
        test_predictions = []
        train_score = 0.0
        test_score = 0.0
        error = str(exc)

    return {
        "problem_id": example.problem_id,
        "train_score": train_score,
        "test_score": test_score,
        "score": test_score,
        "error": error,
        "train_predictions": train_predictions,
        "test_predictions": test_predictions,
        "llm_calls": len(llm.calls),
        "llm_cost": llm.total_cost,
        "llm_traces": llm.calls,
    }


def make_evaluator(*, model: str = MODEL, max_llm_calls: int = 4, profile: str = PROFILE):
    def evaluate(candidate: str, example: ArcExample):
        result = run_candidate(
            candidate,
            example,
            model=model,
            max_llm_calls=max_llm_calls,
            profile=profile,
        )
        side_info = {
            "score": result["score"],
            "problem_id": result["problem_id"],
            "train_score": result["train_score"],
            "test_score": result["test_score"],
            "error": result["error"],
            "llm_calls": result["llm_calls"],
            "llm_cost": result["llm_cost"],
            "llm_traces": result["llm_traces"],
        }
        print(
            f"[{result['problem_id']}] train={result['train_score']:.0%} "
            f"test={result['test_score']:.0%} calls={result['llm_calls']} "
            f"cost=${result['llm_cost']:.4f}",
            flush=True,
        )
        return result["score"], side_info

    return evaluate
