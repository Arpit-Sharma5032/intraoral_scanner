import os

import numpy as np
import pytest

from ioscan.sim.teeth3ds import ENV_VAR, describe, list_cases, load_mesh

pytestmark = pytest.mark.skipif(
    not os.environ.get(ENV_VAR), reason=f"{ENV_VAR} not set; skipping dataset tests"
)


def test_lists_cases():
    assert len(list_cases(jaw="lower")) > 0


def test_load_and_center():
    case = list_cases(jaw="lower")[0]
    mesh = load_mesh(case.obj_path, center=True)
    assert len(mesh.faces) > 1000
    assert np.allclose(mesh.bounds.mean(axis=0), 0.0, atol=1e-6)


def test_units_look_like_millimetres():
    # A dental arch is roughly 40-80 mm across. Catches a wrong-unit dataset early.
    mesh = load_mesh(list_cases(jaw="lower")[0].obj_path)
    ext = describe(mesh)["extents_mm"]
    assert 20 < max(ext) < 150