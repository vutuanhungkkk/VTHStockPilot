"""Promotion gate for candidate ranking models."""
from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationGate:
    minimum_ndcg: float = .72
    minimum_sharpe: float = .8
    maximum_drawdown: float = -.25

    def accepts(self, ndcg: float, sharpe: float, max_drawdown: float) -> bool:
        return ndcg >= self.minimum_ndcg and sharpe >= self.minimum_sharpe and max_drawdown >= self.maximum_drawdown
