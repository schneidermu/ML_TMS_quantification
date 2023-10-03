import pandas as pd
import numpy as np


def max_mass(spectrum: pd.DataFrame) -> float:  # Meaning non-zero mass
    return spectrum[spectrum > 0].index[-1]


def min_mass(spectrum: pd.DataFrame) -> float:  # Meaning non-zero mass
    return spectrum[spectrum > 0].index[0]


def base_peak_intensity(spectrum) -> float:
    return spectrum.max()


def base_peak_mass(spectrum: pd.DataFrame) -> float:
    return np.argmax(spectrum) + 50


def sum_of_intensities(spectrum: pd.Series) -> float:
    return spectrum.sum()


def autocorrelation(spectrum: pd.Series, dm: int) -> float:
    L = max_mass(spectrum)
    F = min_mass(spectrum)
    summ = 0
    for m in range(F, L - dm + 1):
        summ += spectrum[m] * spectrum[m + dm]
    return summ


# Actual features.
def autocorrelation_14_feature(spectrum: pd.DataFrame) -> dict:
    value = 100 * autocorrelation(spectrum, 14) / autocorrelation(spectrum, 0)
    return {'AC-14': value}


def cent_feature(spectrum: pd.DataFrame) -> dict:
    mIm = 100 * spectrum[spectrum > 0].index * spectrum[spectrum > 0]  # m*I
    value = (mIm / (max_mass(spectrum) * sum_of_intensities(spectrum))).sum()
    return {'CENT': value}


def even_feature(spectrum: pd.DataFrame) -> dict:
    I2j = 100 * spectrum[spectrum.index % 2 == 0].sum()
    value = I2j / sum_of_intensities(spectrum)
    return {"EVEN": value}


def mbas_feature(spectrum: pd.DataFrame) -> dict:
    return {"MBAS": base_peak_mass(spectrum)}


def base_feature(spectrum: pd.DataFrame) -> dict:
    value = base_peak_intensity(spectrum) / sum_of_intensities(spectrum) * 100
    return {"BASE": value}


def symx_feature(spectrum: pd.DataFrame) -> dict:
    S = []
    F = min_mass(spectrum)
    L = max_mass(spectrum)
    for m in range(F, L + 1):
        SY = 0
        for j in range(0, min(m - 1, L - m) + 1):
            if m - j >= F and m + j <= L:
                SY += spectrum[m - j] * spectrum[m + j]
        S.append(SY)
    value = 100 / L * (50 + np.argmax(np.array(S)))
    return {"SYMX": value}


def logint1_features(spectrum: pd.DataFrame) -> dict:
    features = []
    masses = range(min_mass(spectrum), max_mass(spectrum))
    for mass in masses:
        value = np.log(max(1, spectrum[mass]) / max(1, spectrum[mass + 1]))
        features.append(value)
    return {f'LG1-{mass}': value for mass, value in zip(masses, features)}


def logint2_features(spectrum: pd.DataFrame) -> dict:
    features = []
    masses = range(min_mass(spectrum), max_mass(spectrum)-1)
    for mass in masses:
        value = np.log(max(1, spectrum[mass]) / max(1, spectrum[mass + 2]))
        features.append(value)
        if value is None:
            print(spectrum[mass], spectrum[mass+2])
    return {f'LG2-{mass}': value for mass, value in zip(masses, features)}


def mod14_features(spectrum: pd.DataFrame) -> dict:
    features = []
    for z in range(1, 15):
        summ = 0
        m_max = max_mass(spectrum)
        for m in range(0, m_max + 1):
            if 50 <= 14 * m + z <= m_max:
                summ += spectrum[14 * m + z]
        features.append(summ)
    features = np.array(features)
    names = [f"MOD14-{i}" for i in range(1, 15)]
    return {key: value for
            key, value in zip(names, features / np.max(features))}


features = [autocorrelation_14_feature,
            cent_feature,
            even_feature,
            mbas_feature,
            base_feature,
            symx_feature,
            logint1_features,
            logint2_features,
            mod14_features]


def generate_features(spectrum: pd.Series) -> pd.Series:
    spectrum = spectrum.loc[50:]
    feature_dict = dict()
    for feature in features:
        try:
            feature_dict.update(feature(spectrum))
        except KeyError:
            print(feature)
    return pd.Series(feature_dict)
