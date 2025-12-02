"""Pytest fixtures and shims for Modal components."""

from __future__ import annotations

import modal


def _build_orders(valid_moves: dict[str, list[str]] | None) -> str:
    orders: list[str] = []
    if valid_moves:
        for unit_moves in valid_moves.values():
            if unit_moves:
                orders.append(unit_moves[0])
    if not orders:
        orders.append("WAIVE")
    return "<orders>\n" + "\n".join(orders) + "\n</orders>"


class _LocalInferenceEngineHandle:
    """Minimal synchronous stub used during local pytest runs."""

    def __call__(self) -> "_LocalInferenceEngineHandle":
        return self

    class _GenerateMethod:
        def remote(
            self,
            prompts: list[str],
            valid_moves: list[dict[str, list[str]]],
            lora_name: str | None = None,
        ) -> list[str]:
            return [_build_orders(moves) for moves in valid_moves]

    generate = _GenerateMethod()


def pytest_configure():
    """Monkey patch Modal lookups so unit tests don't require live infra."""

    original_from_name = modal.Cls.from_name

    def _from_name(app_name: str, cls_name: str):
        if app_name == "diplomacy-grpo" and cls_name == "InferenceEngine":
            return _LocalInferenceEngineHandle()
        return original_from_name(app_name, cls_name)

    modal.Cls.from_name = _from_name
