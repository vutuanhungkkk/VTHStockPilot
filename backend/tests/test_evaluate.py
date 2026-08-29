from app.ml.evaluate import EvaluationGate


def test_candidate_must_pass_every_promotion_gate():
    gate = EvaluationGate()
    assert gate.accepts(ndcg=.8, sharpe=1.1, max_drawdown=-.18)
    assert not gate.accepts(ndcg=.7, sharpe=1.1, max_drawdown=-.18)
    assert not gate.accepts(ndcg=.8, sharpe=.5, max_drawdown=-.18)
