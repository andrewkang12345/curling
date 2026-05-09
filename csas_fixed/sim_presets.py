from __future__ import annotations

CONTACT_MILD_SIM_KWARGS = {
    "c_damp": 165.0,
    "c_damp_sep_frac": 1.0,
    "c_tangent": 20.0,
    "mu_tangent": 0.05,
    "spin_contact": 0.08,
    "k_curl": 0.12,
    "a_linear": 0.10,
    "gamma_spin": 0.12,
}


def contact_mild_params(CurlingParams, **overrides):
    kwargs = dict(CONTACT_MILD_SIM_KWARGS)
    kwargs.update(overrides)
    return CurlingParams(**kwargs)
