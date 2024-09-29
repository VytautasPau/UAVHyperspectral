import colour
import numpy as np
import math


def RGB(array, wavelength, max_size=200_000_000, normalize=False):
    array = hsi_to_xyz(array, wavelength, max_size, normalize)
    
    array = colour.XYZ_to_sRGB(array)

    return array


def hsi_to_xyz(data, wavelength, max_size, normalize=False):
    """
    max_size (int): maximum size of array to integrate. if size > max_size cut array to pieces. 200Mb limit default
    """
    sh = data.shape
    bands = get_bands(wavelength)
    if len(data.shape) > 2:
        data = data[..., np.isin(wavelength, bands)].reshape((sh[0] * sh[1], len(bands)))
    else:
        data = data[:, np.isin(wavelength, bands)]
        
    size = np.prod(data.shape)
    if size > max_size:
        image = None
        for arr in np.array_split(data, math.ceil(size / max_size)):

            if image is None:
                image = integrate(arr, bands)
            else:
                image = np.append(image, integrate(arr, bands), axis=0)
        if normalize:
            image[:, 0] = (image[:, 0] - image[:, 0].min()) / (image[:, 0].max() - image[:, 0].min())
            image[:, 1] = (image[:, 1] - image[:, 1].min()) / (image[:, 1].max() - image[:, 1].min())
            image[:, 2] = (image[:, 2] - image[:, 2].min()) / (image[:, 2].max() - image[:, 2].min())
    else:
        image = integrate(data, bands, normalize)
        
    if len(sh) > 2:
        image = image.reshape((sh[0], sh[1], 3))
    
    return image


def x_cmf(band):
    return colour.colorimetry.wavelength_to_XYZ(band)[:, 0]


def y_cmf(band):
    return colour.colorimetry.wavelength_to_XYZ(band)[:, 1]


def z_cmf(band):
    return colour.colorimetry.wavelength_to_XYZ(band)[:, 2]


def get_bands(wavelength):
    return wavelength[np.logical_and(380 < wavelength, wavelength < 780)]


def integrate(data, bands, normalize=False):
    X = np.trapz(data * x_cmf(bands), bands.reshape((1, -1)), axis=1)
    if normalize:
        image = (X - X.min()) / (X.max() - X.min())
    else:
        image = X
    X = None

    Y = np.trapz(data * y_cmf(bands) * 1.2, bands.reshape((1, -1)), axis=1)
    if normalize:
        image = np.vstack((image, (Y - Y.min()) / (Y.max() - Y.min())))
    else:
        image = np.vstack((image, Y))
    Y = None

    Z = np.trapz(data * z_cmf(bands), bands.reshape((1, -1)), axis=1)
    if normalize:
        image = np.vstack((image, (Z - Z.min()) / (Z.max() - Z.min())))
    else:
        image = np.vstack((image, Z))
    Z = None
    return image.T

