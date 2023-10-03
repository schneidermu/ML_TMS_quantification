import pandas as pd


def extract_data(path: str, base_peak_intensity: float = 999) -> pd.DataFrame:
    current_spectrum = dict()
    masses = set()
    names = set()
    with open(path, "r") as file:
        spectra = []
        for line in file:
            if line.startswith("Name"):
                if current_spectrum:
                    spectra.append(current_spectrum)
                current_spectrum = dict()
            if line[0].isdigit():
                mass, intensity = map(float, line.split())
                current_spectrum[int(mass)] = (intensity
                                               / base_peak_intensity * 100)
                masses.add(int(mass))
            if line[0].isalpha():
                name, value = line.split(": ")
                current_spectrum[name] = value.strip()
                names.add(name)
        spectra.append(current_spectrum)
    mass_features = range(50, max(masses)+1)
    for spectrum in spectra:
        for mass in mass_features:
            spectrum[mass] = spectrum.get(mass, 0)
    names = tuple(names)
    spectra_df = pd.DataFrame(spectra).fillna(0)
    return spectra_df[[*names, *tuple(mass_features)]]
