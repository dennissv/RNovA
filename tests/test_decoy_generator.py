from __future__ import annotations

import pytest

from decoy_spectrum_generator import main as decoy_main
from rnova.mgf import MGFParseError


MGF = """BEGIN IONS
TITLE=sample.1.1.2
PEPMASS=500.0
CHARGE=2+
100.0 10.0
200.0 20.0
300.0 30.0
400.0 40.0
END IONS
"""


def test_decoy_generator_processes_multiple_mgf_files(tmp_path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "decoy"
    input_dir.mkdir()
    (input_dir / "a.mgf").write_text(MGF)
    (input_dir / "b.mgf").write_text(MGF.replace("sample", "other"))

    decoy_main([str(input_dir), str(output_dir)])

    assert (output_dir / "a.mgf.decoy_random0.40.mgf").exists()
    assert (output_dir / "b.mgf.decoy_random0.40.mgf").exists()


def test_decoy_generator_rejects_malformed_mgf(tmp_path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "decoy"
    input_dir.mkdir()
    (input_dir / "bad.mgf").write_text(
        """BEGIN IONS
TITLE=bad.1.1.2
PEPMASS=500.0
CHARGE=2+
100.0 0.0
END IONS
"""
    )

    with pytest.raises(MGFParseError, match="intensity must be positive"):
        decoy_main([str(input_dir), str(output_dir)])
