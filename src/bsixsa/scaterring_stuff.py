import torch
import numpy as np
from kymatio.torch import Scattering1D
from scipy.signal import find_peaks, savgol_filter

def find_peaks_in_spectrum(spectrum, window_length=21, polyorder=3, height=5, distance=20, prominence=5, **kwargs):

    smoothed_spectrum = savgol_filter(spectrum, window_length=window_length, polyorder=polyorder)
    peaks, properties = find_peaks(smoothed_spectrum, height=height, distance=distance, prominence=prominence, **kwargs)

    energies = np.linspace(0, 100, len(spectrum))
    detected_peak_energies = energies[peaks]
    detected_peak_intensities = spectrum[peaks]

    return detected_peak_energies, detected_peak_intensities

def find_peaks_in_spectra(spectra, n_lines=10, find_peak_parameters=None):

    features = []

    if spectra.ndim == 1:
        spectra = spectra[None, :]

    if find_peak_parameters is None:
        find_peak_parameters = {}

    for spectrum in spectra:

        detected_peak_energies_, detected_peak_intensities_ = find_peaks_in_spectrum(spectrum, **find_peak_parameters)

        # Pad to the same length
        if len(detected_peak_energies_) < n_lines:
            detected_peak_energies_ = np.pad(detected_peak_energies_, (0, n_lines - len(detected_peak_energies_)), constant_values=0)
            detected_peak_intensities_ = np.pad(detected_peak_intensities_, (0, n_lines - len(detected_peak_intensities_)), constant_values=0)
        elif len(detected_peak_energies_) > n_lines:
            sorted_indices = np.argsort(detected_peak_intensities_)[-n_lines:]
            detected_peak_energies_ = detected_peak_energies_[sorted_indices]
            detected_peak_intensities_ = detected_peak_intensities_[sorted_indices]

        sorted_indices = np.argsort(detected_peak_energies_)
        detected_peak_energies_ = detected_peak_energies_[sorted_indices]
        detected_peak_intensities_ = detected_peak_intensities_[sorted_indices]

        resulting_features = np.hstack([detected_peak_energies_, detected_peak_intensities_])

        features.append(resulting_features)

    names = [f"Peak {i} energy" for i in range(n_lines)] + [f"Peak {i} intensity" for i in range(n_lines)]

    if spectra.shape[0] == 1:
        return np.vstack(features).squeeze(), names

    return np.vstack(features), names

def scattering_transform(spectra, device="mps"):

    device = torch.device(
        device if
        (device == "mps" and torch.backends.mps.is_available()) or
        (device == "cuda" and torch.cuda.is_available())
        else "cpu"
    )

    T = spectra.shape[-1]
    J = 8
    Q = 12

    scattering = Scattering1D(J, T, Q).to(device)
    spectra = torch.from_numpy(np.asarray(spectra)).to(dtype=torch.float32, device=device)
    coeffs = scattering(spectra)

    return np.array(coeffs.cpu().numpy()), scattering

def compress_with_wst(spectra):
    compressed_coeffs, _ = scattering_transform(spectra)
    return compressed_coeffs.sum(axis=-1), [f"Scattering Coeff {i}" for i in range(compressed_coeffs.shape[1])]

def compress_with_wst_and_search_peaks(spectra):

    features_wst, names_wst = compress_with_wst(spectra)
    features_peaks, names_peaks = find_peaks_in_spectra(spectra)

    res = np.hstack([features_wst, features_peaks])

    return res, names_wst + names_peaks