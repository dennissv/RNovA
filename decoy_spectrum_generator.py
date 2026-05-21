import argparse
import random
from pathlib import Path

from rnova.mgf import read_mgf_blocks


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate decoy MGF files.")
    parser.add_argument("folder", help="Input folder containing .mgf files")
    parser.add_argument("output_folder", help="Output folder for decoy .mgf files")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    folder_path = Path(args.folder)
    output_path = Path(args.output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    mgf_files = sorted(folder_path.glob("*.mgf"))
    if not mgf_files:
        raise SystemExit(f"No .mgf files found in {folder_path}")

    peak_sampling = "random"
    sampling_rate = 0.40
    print("peak_sampling =", peak_sampling)
    print("sampling_rate =", sampling_rate)

    for input_mgf in mgf_files:
        output_mgf = output_path / f"{input_mgf.name}.decoy_{peak_sampling}{sampling_rate:.2f}.mgf"
        _generate_random_decoy(input_mgf, output_mgf, sampling_rate)


def _generate_random_decoy(input_mgf: Path, output_mgf: Path, sampling_rate: float) -> None:
    spectra = read_mgf_blocks(input_mgf)
    peaks_distr = _read_peak_distribution(spectra)
    print("input_mgf =", input_mgf)
    print("output_mgf =", output_mgf)
    print("len(peaks_distr) =", len(peaks_distr))

    sampling_peaks_distr, noise_peaks_distr, decoy_peaks_distr = [], [], []
    with output_mgf.open("w") as f_out:
        for spectrum in spectra:
            f_out.writelines(spectrum.header_lines)
            peak_list = [[peak.moverz, peak.intensity] for peak in spectrum.peaks]
            num_peaks = len(spectrum.peaks)
            num_sampling = int(num_peaks * sampling_rate)
            num_noise = num_peaks - num_sampling
            rng = random.Random(99)
            sampling_peaks = rng.sample(peak_list, num_sampling) if num_sampling else []
            noise_peaks = rng.sample(peaks_distr, num_noise) if num_noise else []

            sampling_peaks_distr += sampling_peaks
            noise_peaks_distr += noise_peaks
            decoy_peaks_distr += sampling_peaks + noise_peaks

            for x, y in sorted(sampling_peaks + noise_peaks, key=lambda peak: peak[0]):
                f_out.write("{0:.5f} {1:.5f}\n".format(x, y))
            f_out.write(spectrum.end_line)

    print("len(sampling_peaks_distr) =", len(sampling_peaks_distr))
    print("len(noise_peaks_distr) =", len(noise_peaks_distr))
    print("len(decoy_peaks_distr) =", len(decoy_peaks_distr))
    print()


def _read_peak_distribution(spectra) -> list[list[float]]:
    peaks: list[list[float]] = []
    for spectrum in spectra:
        for peak in spectrum.peaks:
            peaks.append([peak.moverz, peak.intensity])
    return peaks

if __name__ == "__main__":
    main()
