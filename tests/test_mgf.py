from __future__ import annotations

import math

import pytest

from rnova.mgf import MGFParseError, extract_scan_id, read_mgf


def test_extract_scan_id_prefers_scans() -> None:
    assert extract_scan_id(title="file.1.2.3", scans="99", spectrum_index=1) == "99"


def test_extract_scan_id_from_native_id_scan() -> None:
    title = (
        'S00465_Exploris02_BB_Bovatus_CPLGlu_1.7.7.2 '
        'File:"S00465_Exploris02_BB_Bovatus_CPLGlu_1.raw", '
        'NativeID:"controllerType=0 controllerNumber=1 scan=7"'
    )
    assert extract_scan_id(title=title, spectrum_index=1) == "7"


def test_extract_scan_id_from_dot_title() -> None:
    assert extract_scan_id(title="sample.123.123.2", spectrum_index=1) == "123"


def test_extract_scan_id_falls_back_to_index() -> None:
    assert extract_scan_id(spectrum_index=4) == "4"


def test_read_mgf_handles_native_id_title_and_whitespace_peaks(tmp_path) -> None:
    mgf = tmp_path / "input.mgf"
    mgf.write_text(
        "\n".join(
            [
                "BEGIN IONS",
                'TITLE=S00465_Exploris02_BB_Bovatus_CPLGlu_1.7.7.2 File:"x.raw", NativeID:"controllerType=0 controllerNumber=1 scan=7"',
                "RTINSECONDS=2.353758122",
                "PEPMASS=434.8857421875 47216.063842773438",
                "CHARGE=2+",
                "100.0   200.0",
                "150.5\t300.5",
                "END IONS",
                "",
            ]
        )
    )

    spectra = read_mgf(mgf)

    assert len(spectra) == 1
    assert spectra[0]["title"] == "7"
    assert spectra[0]["precursor_moverz"] == 434.8857421875
    assert spectra[0]["precursor_charge"] == 2
    assert spectra[0]["product_ion_moverz"].tolist() == [100.0, 150.5]
    assert math.isclose(spectra[0]["product_ion_intensity_log"][0], math.log(200.0))


def test_read_mgf_reports_missing_required_fields(tmp_path) -> None:
    mgf = tmp_path / "bad.mgf"
    mgf.write_text(
        "\n".join(
            [
                "BEGIN IONS",
                "TITLE=file.1.1.2",
                "CHARGE=2+",
                "100.0 200.0",
                "END IONS",
            ]
        )
    )

    with pytest.raises(MGFParseError, match="missing PEPMASS"):
        read_mgf(mgf)


def test_read_mgf_rejects_zero_intensity(tmp_path) -> None:
    mgf = tmp_path / "zero-intensity.mgf"
    mgf.write_text(
        "\n".join(
            [
                "BEGIN IONS",
                "TITLE=file.1.1.2",
                "PEPMASS=500.0",
                "CHARGE=2+",
                "100.0 0.0",
                "END IONS",
            ]
        )
    )

    with pytest.raises(MGFParseError, match="intensity must be positive"):
        read_mgf(mgf)


def test_read_mgf_rejects_empty_spectrum(tmp_path) -> None:
    mgf = tmp_path / "empty.mgf"
    mgf.write_text(
        "\n".join(
            [
                "BEGIN IONS",
                "TITLE=file.1.1.2",
                "PEPMASS=500.0",
                "CHARGE=2+",
                "END IONS",
            ]
        )
    )

    with pytest.raises(MGFParseError, match="contains no peaks"):
        read_mgf(mgf)


def test_read_mgf_rejects_duplicate_begin(tmp_path) -> None:
    mgf = tmp_path / "duplicate-begin.mgf"
    mgf.write_text(
        "\n".join(
            [
                "BEGIN IONS",
                "TITLE=file.1.1.2",
                "BEGIN IONS",
            ]
        )
    )

    with pytest.raises(MGFParseError, match="BEGIN IONS before previous spectrum ended"):
        read_mgf(mgf)


def test_read_mgf_rejects_missing_end(tmp_path) -> None:
    mgf = tmp_path / "missing-end.mgf"
    mgf.write_text(
        "\n".join(
            [
                "BEGIN IONS",
                "TITLE=file.1.1.2",
                "PEPMASS=500.0",
                "CHARGE=2+",
                "100.0 10.0",
            ]
        )
    )

    with pytest.raises(MGFParseError, match="missing END IONS"):
        read_mgf(mgf)


def test_read_mgf_rejects_malformed_charge(tmp_path) -> None:
    mgf = tmp_path / "bad-charge.mgf"
    mgf.write_text(
        "\n".join(
            [
                "BEGIN IONS",
                "TITLE=file.1.1.2",
                "PEPMASS=500.0",
                "CHARGE=two+",
                "100.0 10.0",
                "END IONS",
            ]
        )
    )

    with pytest.raises(MGFParseError, match="failed to parse line"):
        read_mgf(mgf)


def test_read_mgf_handles_mixed_case_headers(tmp_path) -> None:
    mgf = tmp_path / "mixed.mgf"
    mgf.write_text(
        "\n".join(
            [
                "begin ions",
                "title=file.4.4.2",
                "pepmass=500.0",
                "charge=2+",
                "scans=42",
                "100.0 10.0",
                "end ions",
            ]
        )
    )

    spectra = read_mgf(mgf)

    assert spectra[0]["title"] == "42"


def test_read_mgf_rejects_malformed_peak_line(tmp_path) -> None:
    mgf = tmp_path / "bad-peak.mgf"
    mgf.write_text(
        "\n".join(
            [
                "BEGIN IONS",
                "TITLE=file.1.1.2",
                "PEPMASS=500.0",
                "CHARGE=2+",
                "100.0",
                "END IONS",
            ]
        )
    )

    with pytest.raises(MGFParseError, match="peak line must contain"):
        read_mgf(mgf)
