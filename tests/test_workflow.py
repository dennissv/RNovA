from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")

from src.fill_PTM import UniModMassIndex
from workflow import delta_mass_to_ptm


def test_unimod_topk_handoff_uses_seqfiller_mass_token() -> None:
    filled_result = pd.DataFrame([{"filled_sequence": "AT(+57.021)G"}])
    index = UniModMassIndex(
        [
            {
                "title": "Carbamidomethyl",
                "unimod_id": 4,
                "mono_mass": 57.021464,
                "specificity": [("T", "Anywhere")],
            }
        ]
    )

    annotated, ptm_freq = delta_mass_to_ptm(filled_result, index, k=1, tol=0.01)

    assert ptm_freq == {"T[57.021464]": 1}
    assert annotated.loc[0, "mod_residue"] == "T2"
    assert annotated.loc[0, "mod_name"] == "Carbamidomethyl|UniMod:4"
