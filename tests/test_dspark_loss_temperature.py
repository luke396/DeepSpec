"""Regression tests for the optional DSpark loss temperature (issue #253).

The scalar temperature only rescales the CE and L1 terms as
``T * f(z / T)``. Confidence targets and the train-side accept-rate metrics
stay on the frozen T=1 definition, and ``loss_temperature=None`` must remain
bit-for-bit identical to the historical code path.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from deepspec.modeling.dspark import loss as loss_module
from deepspec.modeling.dspark.common import DSparkForwardOutput
from deepspec.modeling.dspark.loss import (
    _collect_local_terms,
    compute_dspark_loss,
)
from deepspec.utils import metrics as metrics_module

LOSS_TERM_KEYS = (
    "ce_loss_num",
    "ce_loss_den",
    "l1_loss_num",
    "l1_loss_den",
    "confidence_loss_num",
    "confidence_loss_den",
)


@pytest.fixture(autouse=True)
def _reset_metrics():
    metrics_module.reset()
    yield
    metrics_module.reset()


def _make_random_outputs(*, with_confidence=True, seed=0):
    generator = torch.Generator().manual_seed(seed)
    batch, num_anchors, block_size, vocab = 2, 2, 3, 5
    draft_logits = torch.randn(
        batch, num_anchors, block_size, vocab, generator=generator
    )
    aligned_target_logits = torch.randn(
        batch, num_anchors, block_size, vocab, generator=generator
    )
    target_ids = torch.randint(
        0, vocab, (batch, num_anchors, block_size), generator=generator
    )
    eval_mask = torch.ones(batch, num_anchors, block_size, dtype=torch.bool)
    eval_mask[0, 1, 2] = False
    eval_mask[1, 0, 1:] = False
    block_keep_mask = torch.ones(batch, num_anchors, dtype=torch.bool)
    block_keep_mask[1, 1] = False
    confidence_pred = None
    if with_confidence:
        confidence_pred = torch.randn(
            batch, num_anchors, block_size, generator=generator
        )
    return DSparkForwardOutput(
        draft_logits=draft_logits,
        target_ids=target_ids,
        eval_mask=eval_mask,
        block_keep_mask=block_keep_mask,
        confidence_pred=confidence_pred,
        aligned_target_logits=aligned_target_logits,
    )


def _collect(outputs, **kwargs):
    kwargs.setdefault("loss_decay_gamma", None)
    kwargs.setdefault("l1_loss_alpha", 0.9)
    loss_terms, _ = _collect_local_terms(outputs=outputs, **kwargs)
    return loss_terms


def _golden_pre_change_terms(outputs, *, loss_decay_gamma):
    """Pre-#253 loss terms, written out with the historical T=1 formulas."""
    draft_logits = outputs.draft_logits
    _, _, block_size, vocab = draft_logits.shape

    weights = outputs.eval_mask.to(torch.float32)
    if loss_decay_gamma is not None and loss_decay_gamma > 0:
        positions = torch.arange(block_size).view(1, 1, -1)
        weights = weights * torch.exp(-positions.float() / float(loss_decay_gamma))

    flat_weights = weights.reshape(-1)
    loss_per_token = F.cross_entropy(
        draft_logits.reshape(-1, vocab),
        outputs.target_ids.reshape(-1),
        reduction="none",
    )
    ce_loss_num = (loss_per_token * flat_weights).sum()
    ce_loss_den = flat_weights.sum()

    draft_probs = torch.softmax(draft_logits.float(), dim=-1)
    target_probs = torch.softmax(outputs.aligned_target_logits.float(), dim=-1)
    l1_dist = (draft_probs - target_probs).abs().sum(dim=-1)
    l1_loss_num = (l1_dist * weights).sum()
    l1_loss_den = weights.sum()

    accept_rate = (1.0 - 0.5 * l1_dist).clamp(0.0, 1.0)
    confidence_errors = F.binary_cross_entropy_with_logits(
        outputs.confidence_pred.float(),
        accept_rate.detach(),
        reduction="none",
    ) * weights
    return {
        "ce_loss_num": ce_loss_num,
        "ce_loss_den": ce_loss_den,
        "l1_loss_num": l1_loss_num,
        "l1_loss_den": l1_loss_den,
        "confidence_loss_num": confidence_errors.sum(),
        "confidence_loss_den": weights.sum(),
    }


@pytest.mark.parametrize("loss_decay_gamma", [None, 4.0])
def test_none_temperature_matches_pre_change_golden(loss_decay_gamma):
    outputs = _make_random_outputs()
    golden = _golden_pre_change_terms(outputs, loss_decay_gamma=loss_decay_gamma)

    explicit_none = _collect(
        outputs, loss_decay_gamma=loss_decay_gamma, loss_temperature=None
    )
    default_kwarg = _collect(outputs, loss_decay_gamma=loss_decay_gamma)

    for key in LOSS_TERM_KEYS:
        assert torch.equal(explicit_none[key], golden[key]), key
        assert torch.equal(default_kwarg[key], golden[key]), key


def _make_toy_outputs():
    """Two supervised tokens over a 3-token vocab, weights all 1."""
    draft_logits = torch.tensor(
        [[[[2.0, 0.0, -1.0], [0.5, -0.5, 1.5]]]], dtype=torch.float32
    )
    aligned_target_logits = torch.tensor(
        [[[[1.0, 0.5, 0.0], [0.0, 0.0, 1.0]]]], dtype=torch.float32
    )
    target_ids = torch.tensor([[[0, 2]]], dtype=torch.long)
    return DSparkForwardOutput(
        draft_logits=draft_logits,
        target_ids=target_ids,
        eval_mask=torch.ones(1, 1, 2, dtype=torch.bool),
        block_keep_mask=torch.ones(1, 1, dtype=torch.bool),
        confidence_pred=None,
        aligned_target_logits=aligned_target_logits,
    )


def test_scalar_temperature_matches_hand_computed_values():
    # Hand computation at T=0.7 (float64):
    #   token 0: z_d=[2,0,-1], y=0, z_t=[1,.5,0]
    #   token 1: z_d=[.5,-.5,1.5], y=2, z_t=[0,0,1]
    #   ce_i = 0.7 * (logsumexp(z_d/0.7) - z_d[y]/0.7)
    #     ce_0 = 0.048143312191843840, ce_1 = 0.182082881851321460
    #   l1_i = 0.7 * sum_j |softmax(z_d/0.7)_j - softmax(z_t/0.7)_j|
    #     l1_0 = 0.497323499846069000, l1_1 = 0.164814303944773980
    outputs = _make_toy_outputs()
    loss_terms = _collect(outputs, loss_temperature=0.7)

    assert loss_terms["ce_loss_num"].item() == pytest.approx(
        0.2302261940431653, abs=1e-6
    )
    assert loss_terms["ce_loss_den"].item() == pytest.approx(2.0)
    assert loss_terms["l1_loss_num"].item() == pytest.approx(
        0.662137803790843, abs=1e-6
    )
    assert loss_terms["l1_loss_den"].item() == pytest.approx(2.0)

    # Cross-check the hand-derived constants against float64 torch math.
    z_d = outputs.draft_logits.double() / 0.7
    z_t = outputs.aligned_target_logits.double() / 0.7
    expected_ce = 0.7 * F.cross_entropy(
        z_d.reshape(-1, 3), outputs.target_ids.reshape(-1), reduction="none"
    )
    assert expected_ce.sum().item() == pytest.approx(0.2302261940431653, abs=1e-12)
    expected_l1 = 0.7 * (
        (torch.softmax(z_d, dim=-1) - torch.softmax(z_t, dim=-1)).abs().sum(dim=-1)
    )
    assert expected_l1.sum().item() == pytest.approx(0.662137803790843, abs=1e-12)


def test_temperature_one_matches_none_bitwise():
    outputs = _make_random_outputs()
    none_terms = _collect(outputs, loss_temperature=None)
    one_terms = _collect(outputs, loss_temperature=1.0)
    for key in LOSS_TERM_KEYS:
        assert torch.equal(one_terms[key], none_terms[key]), key


def test_confidence_and_accept_metrics_are_temperature_invariant(monkeypatch):
    captured = {}

    def _capture_factory(store):
        def _capture(name, value, *, den=None, reduction=None, tag="train"):
            store[f"{tag}/{name}"] = (
                value.detach().clone(),
                None if den is None else den.detach().clone(),
            )

        return _capture

    def _run(loss_temperature):
        outputs = _make_random_outputs(seed=7)
        store = {}
        monkeypatch.setattr(loss_module, "add_metric", _capture_factory(store))
        terms = _collect(outputs, loss_temperature=loss_temperature)
        return terms, store

    none_terms, none_metrics = _run(None)
    for loss_temperature in (0.7,):
        temp_terms, temp_metrics = _run(loss_temperature)

        # CE/L1 must actually move, otherwise this test proves nothing.
        assert not torch.equal(temp_terms["ce_loss_num"], none_terms["ce_loss_num"])
        assert not torch.equal(temp_terms["l1_loss_num"], none_terms["l1_loss_num"])

        # Confidence target/loss stay on the frozen T=1 definition.
        assert torch.equal(
            temp_terms["confidence_loss_num"], none_terms["confidence_loss_num"]
        )
        assert torch.equal(
            temp_terms["confidence_loss_den"], none_terms["confidence_loss_den"]
        )

        # accept_rate@k / tau_probabilistic / confidence_* metrics keep the
        # T=1 numbers so wandb curves stay comparable with the A arm.
        assert set(temp_metrics) == set(none_metrics)
        for name, (num, den) in none_metrics.items():
            assert torch.equal(temp_metrics[name][0], num), name
            assert (den is None) == (temp_metrics[name][1] is None), name
            if den is not None:
                assert torch.equal(temp_metrics[name][1], den), name


@pytest.mark.parametrize(
    "loss_temperature",
    [0.0, -0.5],
    ids=["zero", "negative"],
)
def test_non_positive_temperature_raises(loss_temperature):
    outputs = _make_toy_outputs()
    with pytest.raises(AssertionError, match="loss_temperature"):
        _collect(outputs, loss_temperature=loss_temperature)


def test_compute_dspark_loss_threads_temperature(monkeypatch):
    monkeypatch.setattr(loss_module.dist, "get_world_size", lambda: 1)

    def _loss(loss_temperature):
        metrics_module.reset()
        outputs = _make_random_outputs(seed=3)
        return compute_dspark_loss(
            outputs=outputs,
            loss_decay_gamma=4.0,
            ce_loss_alpha=0.1,
            l1_loss_alpha=0.9,
            confidence_head_alpha=1.0,
            loss_temperature=loss_temperature,
        )

    baseline = _loss(None)
    assert torch.equal(_loss(1.0), baseline)
    assert not torch.equal(_loss(0.7), baseline)

    # Sanity: the T->0 limit of T * CE(z/T, y) is the margin loss
    # max_j z_j - z_y, so the scaled loss stays finite for small T.
    small_t = _loss(1e-3)
    assert torch.isfinite(small_t)


def test_per_sample_tensor_constant_matches_scalar_bitwise():
    outputs = _make_random_outputs()
    scalar_terms = _collect(outputs, loss_temperature=0.7)
    tensor_terms = _collect(
        outputs,
        loss_temperature=torch.full((2,), 0.7, dtype=torch.float32),
    )
    for key in LOSS_TERM_KEYS:
        assert torch.equal(tensor_terms[key], scalar_terms[key]), key


def test_per_sample_tensor_mixes_per_row_scalars():
    # A [batch] tensor must equal computing each sample with its own scalar
    # temperature and summing the (numerator, denominator) pairs.
    outputs = _make_random_outputs()
    temperatures = (0.5, 1.2)
    mixed_terms = _collect(
        outputs,
        loss_temperature=torch.tensor(temperatures, dtype=torch.float32),
    )

    def _single(sample_index):
        sliced = DSparkForwardOutput(
            draft_logits=outputs.draft_logits[sample_index : sample_index + 1],
            target_ids=outputs.target_ids[sample_index : sample_index + 1],
            eval_mask=outputs.eval_mask[sample_index : sample_index + 1],
            block_keep_mask=outputs.block_keep_mask[sample_index : sample_index + 1],
            confidence_pred=outputs.confidence_pred[sample_index : sample_index + 1],
            aligned_target_logits=outputs.aligned_target_logits[
                sample_index : sample_index + 1
            ],
        )
        return _collect(sliced, loss_temperature=temperatures[sample_index])

    first, second = _single(0), _single(1)
    for key in LOSS_TERM_KEYS:
        combined = first[key] + second[key]
        assert torch.allclose(mixed_terms[key], combined, atol=1e-5), key


def test_per_sample_tensor_keeps_confidence_and_metrics_frozen(monkeypatch):
    captured_none, captured_tensor = {}, {}

    def _capture_factory(store):
        def _capture(name, value, *, den=None, reduction=None, tag="train"):
            store[f"{tag}/{name}"] = (
                value.detach().clone(),
                None if den is None else den.detach().clone(),
            )

        return _capture

    outputs = _make_random_outputs(seed=11)
    monkeypatch.setattr(loss_module, "add_metric", _capture_factory(captured_none))
    none_terms = _collect(outputs, loss_temperature=None)

    outputs = _make_random_outputs(seed=11)
    monkeypatch.setattr(loss_module, "add_metric", _capture_factory(captured_tensor))
    tensor_terms = _collect(
        outputs,
        loss_temperature=torch.tensor([0.5, 1.2], dtype=torch.float32),
    )

    assert not torch.equal(tensor_terms["ce_loss_num"], none_terms["ce_loss_num"])
    assert not torch.equal(tensor_terms["l1_loss_num"], none_terms["l1_loss_num"])
    assert torch.equal(
        tensor_terms["confidence_loss_num"], none_terms["confidence_loss_num"]
    )
    assert set(captured_tensor) == set(captured_none)
    for name, (num, den) in captured_none.items():
        assert torch.equal(captured_tensor[name][0], num), name
        if den is not None:
            assert torch.equal(captured_tensor[name][1], den), name


def test_compute_dspark_loss_accepts_per_sample_tensor(monkeypatch):
    monkeypatch.setattr(loss_module.dist, "get_world_size", lambda: 1)
    outputs = _make_random_outputs()
    loss = compute_dspark_loss(
        outputs=outputs,
        loss_decay_gamma=None,
        ce_loss_alpha=0.1,
        l1_loss_alpha=0.9,
        confidence_head_alpha=1.0,
        loss_temperature=torch.tensor([0.5, 1.2], dtype=torch.float32),
    )
    assert torch.isfinite(loss)
