from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator
import numpy as np
from matplotlib import pyplot as plt
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import torch
from scipy.signal import find_peaks


def generate_xrd(structure, angle_min, angle_max, dim, fwhm):
    xrd_calculator = XRDCalculator(wavelength="CuKa")
    xrd_pattern = xrd_calculator.get_pattern(structure)

    return gaussian_convolute(xrd_pattern, angle_min, angle_max, dim, fwhm)


def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def gaussian_convolute(xrd_pattern, angle_min, angle_max, dim, fwhm):
    x_vals = np.linspace(angle_min, angle_max, dim)
    y_vals = np.zeros_like(x_vals)

    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))

    for peak_pos, intensity in zip(xrd_pattern.x, xrd_pattern.y):
        y_vals += intensity * gaussian(x_vals, peak_pos, sigma)
    y_vals = y_vals/np.max(y_vals)

    return x_vals, y_vals, torch.tensor(xrd_pattern.x), torch.tensor(xrd_pattern.y)/100


def generate_gaussian_convolution_mat(angle, fwhm, angle_min, angle_max, dim):
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    angle_all = np.linspace(angle_min, angle_max, dim)

    gaussian_convolution_mat = np.array(
        [gaussian(angle_all, np.array(angle[i]), sigma) for i in range(angle.shape[0])]).T

    return gaussian_convolution_mat