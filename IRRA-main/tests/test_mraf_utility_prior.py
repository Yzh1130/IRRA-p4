import pytest
import torch
import torch.nn.functional as F

from model.mraf import compute_mraf_reliability_targets


def _inputs():
    clean_rgb = torch.tensor(
        [[1.0, 0.0, 0.0], [0.5, 0.5, 0.0]], dtype=torch.float32
    )
    clean_ir = torch.tensor(
        [[0.0, 1.0, 0.0], [0.0, 0.5, 0.5]], dtype=torch.float32
    )
    degraded_rgb = torch.tensor(
        [[0.8, 0.2, 0.0], [0.4, 0.6, 0.0]], dtype=torch.float32
    )
    degraded_ir = torch.tensor(
        [[0.1, 0.9, 0.0], [0.0, 0.3, 0.7]], dtype=torch.float32
    )
    return clean_rgb, clean_ir, degraded_rgb, degraded_ir


def _legacy_outputs(clean_rgb, clean_ir, degraded_rgb, degraded_ir, temperature):
    drift_rgb = 1.0 - F.cosine_similarity(
        degraded_rgb.float(), clean_rgb.detach().float(), dim=-1, eps=1e-8
    )
    drift_ir = 1.0 - F.cosine_similarity(
        degraded_ir.float(), clean_ir.detach().float(), dim=-1, eps=1e-8
    )
    target_logits = torch.stack([-drift_rgb, -drift_ir], dim=-1)
    target = F.softmax(target_logits / float(temperature), dim=-1).detach()
    return {
        "target": target,
        "target_rgb": target[:, 0:1],
        "target_ir": target[:, 1:2],
        "drift_rgb": drift_rgb,
        "drift_ir": drift_ir,
    }


def test_none_and_unit_utility_prior_are_bitwise_legacy_equivalent():
    inputs = _inputs()
    temperature = 0.3
    legacy = _legacy_outputs(*inputs, temperature)
    no_prior = compute_mraf_reliability_targets(
        *inputs, target_temperature=temperature
    )
    unit_prior = compute_mraf_reliability_targets(
        *inputs,
        target_temperature=temperature,
        utility_prior=[1.0, 1.0],
    )

    for key, expected in legacy.items():
        assert torch.equal(no_prior[key], expected)
        assert torch.equal(unit_prior[key], expected)
    assert torch.equal(no_prior["utility_prior"], torch.ones(2))
    assert torch.equal(unit_prior["utility_prior"], torch.ones(2))


def test_rgb_utility_prior_monotonically_increases_rgb_target():
    inputs = _inputs()
    baseline = compute_mraf_reliability_targets(
        *inputs, target_temperature=0.3
    )["target_rgb"]
    outputs = compute_mraf_reliability_targets(
        *inputs,
        target_temperature=0.3,
        utility_prior=[2.0, 1.0],
    )
    expected_logits = (
        torch.stack([-outputs["drift_rgb"], -outputs["drift_ir"]], dim=-1)
        / 0.3
        + torch.tensor([2.0, 1.0]).log()
    )

    assert torch.equal(outputs["target"], F.softmax(expected_logits, dim=-1))
    assert torch.equal(outputs["utility_prior"], torch.tensor([2.0, 1.0]))
    assert bool((outputs["target_rgb"] > baseline).all().item())


@pytest.mark.parametrize(
    "utility_prior",
    ([0.0, 1.0], [-1.0, 1.0], [1.0, 0.0], [1.0, float("nan")]),
)
def test_nonpositive_or_nonfinite_utility_prior_raises(utility_prior):
    with pytest.raises(ValueError, match="utility_prior"):
        compute_mraf_reliability_targets(
            *_inputs(),
            target_temperature=0.3,
            utility_prior=utility_prior,
        )


@pytest.mark.parametrize("utility_prior", ([1.0], [1.0, 1.0, 1.0]))
def test_utility_prior_must_have_two_values(utility_prior):
    with pytest.raises(ValueError, match="length-2"):
        compute_mraf_reliability_targets(
            *_inputs(),
            target_temperature=0.3,
            utility_prior=utility_prior,
        )
