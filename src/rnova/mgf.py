from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


class MGFParseError(ValueError):
    """Raised when an MGF spectrum cannot be parsed safely."""


@dataclass
class MGFPeak:
    moverz: float
    intensity: float


@dataclass
class MGFSpectrumBlock:
    index: int
    title: str | None
    scans: str | None
    precursor_moverz: float
    charge: int
    peaks: tuple[MGFPeak, ...]
    header_lines: tuple[str, ...]
    end_line: str = "END IONS\n"


@dataclass
class _SpectrumState:
    index: int
    start_line: int
    charge: int | None = None
    title: str | None = None
    scans: str | None = None
    precursor_moverz: float | None = None
    peaks: list[MGFPeak] | None = None
    header_lines: list[str] | None = None

    def __post_init__(self) -> None:
        self.peaks = []
        self.header_lines = ["BEGIN IONS\n"]


def extract_scan_id(
    title: str | None = None,
    scans: str | None = None,
    spectrum_index: int | None = None,
) -> str:
    if scans:
        return scans.strip()

    if title and "scan=" in title:
        return title.split("scan=", 1)[1].split('"', 1)[0].split()[0]

    if title:
        first_token = title.split(None, 1)[0]
        parts = first_token.split(".")
        if len(parts) >= 3:
            return parts[-3]
        if len(parts) >= 2:
            return parts[-2]
        return first_token

    if spectrum_index is not None:
        return str(spectrum_index)

    return "0"


def read_mgf(mgf_file: str | Path, default_charge: int | None = None) -> list[dict]:
    spectra = []
    for block in read_mgf_blocks(mgf_file, default_charge=default_charge):
        spectra.append(
            {
                "title": extract_scan_id(
                    title=block.title,
                    scans=block.scans,
                    spectrum_index=block.index,
                ),
                "precursor_moverz": block.precursor_moverz,
                "precursor_charge": block.charge,
                "product_ion_moverz": np.array([peak.moverz for peak in block.peaks]),
                "product_ion_intensity_log": np.log(
                    np.array([peak.intensity for peak in block.peaks])
                ),
            }
        )
    return spectra


def read_mgf_blocks(mgf_file: str | Path, default_charge: int | None = None) -> list[MGFSpectrumBlock]:
    path = Path(mgf_file)
    spectra: list[MGFSpectrumBlock] = []
    current: _SpectrumState | None = None
    spectrum_index = 0

    with path.open() as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            upper_line = line.upper()
            if not line:
                continue

            if upper_line == "BEGIN IONS":
                if current is not None:
                    raise _parse_error(
                        path,
                        line_number,
                        current.index,
                        "found BEGIN IONS before previous spectrum ended",
                    )
                spectrum_index += 1
                current = _SpectrumState(
                    index=spectrum_index,
                    start_line=line_number,
                    charge=default_charge,
                )
                continue

            if current is None:
                if upper_line == "END IONS":
                    raise _parse_error(path, line_number, None, "found END IONS before BEGIN IONS")
                raise _parse_error(path, line_number, None, f"unexpected line outside spectrum: {line!r}")

            try:
                if upper_line == "END IONS":
                    spectra.append(_finish_spectrum(path, current, raw_line))
                    current = None
                elif "=" in line:
                    _parse_header_line(path, line_number, current, raw_line, line)
                elif _looks_like_peak(line):
                    current.peaks.append(_parse_peak_line(path, line_number, current.index, line))
                else:
                    raise _parse_error(
                        path,
                        line_number,
                        current.index,
                        f"unexpected line in spectrum: {line!r}",
                    )
            except MGFParseError:
                raise
            except (IndexError, TypeError, ValueError) as exc:
                raise _parse_error(
                    path,
                    line_number,
                    current.index,
                    f"failed to parse line: {line!r}",
                ) from exc

    if current is not None:
        raise MGFParseError(
            f"{path}: spectrum {current.index} beginning at line {current.start_line} is missing END IONS"
        )

    return spectra


def _parse_header_line(
    path: Path,
    line_number: int,
    current: _SpectrumState,
    raw_line: str,
    line: str,
) -> None:
    key, raw_value = _split_header(line)
    current.header_lines.append(raw_line)
    if key == "TITLE":
        current.title = raw_value
    elif key == "PEPMASS":
        parts = raw_value.split()
        if not parts:
            raise _parse_error(path, line_number, current.index, "PEPMASS is empty")
        current.precursor_moverz = float(parts[0])
    elif key == "CHARGE":
        current.charge = _parse_charge(raw_value)
    elif key == "SCANS":
        current.scans = raw_value


def _split_header(line: str) -> tuple[str, str]:
    if "=" not in line:
        raise ValueError("expected key=value line")
    key, value = line.split("=", 1)
    return key.strip().upper(), value.strip()


def _parse_charge(raw_charge: str) -> int:
    charge = raw_charge.strip()
    if not charge:
        raise ValueError("empty charge")
    if charge.endswith("+"):
        charge = charge[:-1]
        sign = 1
    elif charge.endswith("-"):
        charge = charge[:-1]
        sign = -1
    else:
        sign = 1
    if not charge.isdigit():
        raise ValueError("malformed charge")
    value = int(charge) * sign
    if value == 0:
        raise ValueError("zero charge")
    return value


def _looks_like_peak(line: str) -> bool:
    first = line[:1]
    return first.isdigit() or first in {".", "+", "-"}


def _parse_peak_line(path: Path, line_number: int, spectrum_index: int, line: str) -> MGFPeak:
    parts = line.split()
    if len(parts) < 2:
        raise _parse_error(path, line_number, spectrum_index, "peak line must contain m/z and intensity")
    moverz = float(parts[0])
    intensity = float(parts[1])
    if moverz <= 0:
        raise _parse_error(path, line_number, spectrum_index, f"peak m/z must be positive: {moverz}")
    if intensity <= 0:
        raise _parse_error(path, line_number, spectrum_index, f"peak intensity must be positive: {intensity}")
    return MGFPeak(moverz=moverz, intensity=intensity)


def _finish_spectrum(path: Path, spectrum: _SpectrumState, end_line: str) -> MGFSpectrumBlock:
    if spectrum.precursor_moverz is None:
        raise MGFParseError(
            f"{path}: spectrum {spectrum.index} is missing PEPMASS"
        )
    if spectrum.charge is None:
        raise MGFParseError(
            f"{path}: spectrum {spectrum.index} is missing CHARGE"
        )
    if spectrum.precursor_moverz <= 0:
        raise MGFParseError(
            f"{path}: spectrum {spectrum.index} PEPMASS must be positive"
        )
    if not spectrum.peaks:
        raise MGFParseError(
            f"{path}: spectrum {spectrum.index} contains no peaks"
        )

    return MGFSpectrumBlock(
        index=spectrum.index,
        title=spectrum.title,
        scans=spectrum.scans,
        precursor_moverz=spectrum.precursor_moverz,
        charge=spectrum.charge,
        peaks=tuple(spectrum.peaks),
        header_lines=tuple(spectrum.header_lines or ()),
        end_line=end_line if end_line.endswith("\n") else f"{end_line}\n",
    )


def _parse_error(path: Path, line_number: int, spectrum_index: int | None, message: str) -> MGFParseError:
    spectrum = f"spectrum {spectrum_index}: " if spectrum_index is not None else ""
    return MGFParseError(f"{path}:{line_number}: {spectrum}{message}")
