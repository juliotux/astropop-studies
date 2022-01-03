import numpy as np
from astropy.utils import NumpyRNGContext
from astropop.math.moffat import moffat_2d
from astropop.math.gaussian import gaussian_2d
from astropop.py_utils import check_number


def gen_bkg(size, level, rdnoise, rng_seed=123, dtype='f8'):
    """Generate a simple background image."""
    # create a level image
    im = np.ones(size, dtype)*level

    # reate the gaussian read noise image to sum
    with NumpyRNGContext(rng_seed):
        noise = np.random.normal(loc=0, scale=rdnoise, size=size)
    im += noise

    # poisonic not needed?
    return im


def gen_position_flux(size, number, low, high, rng_seed=123):
    """Generate x, y, and flux lists for stars."""
    for i in range(number):
        with NumpyRNGContext(rng_seed):
            x = np.random.randint(0, size[0], number)
        with NumpyRNGContext(rng_seed+i):
            y = np.random.randint(0, size[1], number)
            flux = np.random.randint(low, high, number)
    return x, y, flux


def gen_stars_moffat(size, x, y, flux, fwhm):
    """Generate stars image to add to background."""
    beta = 1.5
    alpha = fwhm/np.sqrt(2**(1/beta)-1)
    im = np.zeros(size)
    grid_y, grid_x = np.indices(size)
    for xi, yi, fi in zip(x, y, flux):
        im += moffat_2d(grid_x, grid_y, xi, yi, alpha, beta, fi, 0)

    return im


def gen_stars_gaussian(size, x, y, flux, sigma, theta):
    """Generate stars image to add to background."""
    im = np.zeros(size)
    grid_y, grid_x = np.indices(size)

    try:
        sigma_x, sigma_y = sigma
    except:
        sigma_x = sigma_y = sigma

    if check_number(sigma_x):
        sigma_x = [sigma_x]*len(x)

    if check_number(sigma_y):
        sigma_y = [sigma_y]*len(x)

    if check_number(theta):
        theta = [theta]*len(x)

    for xi, yi, fi, sxi, syi, ti in zip(x, y, flux, sigma_x, sigma_y, theta):
        im += gaussian_2d(grid_x, grid_y, xi, yi, sxi, syi, ti, fi, 0)

    return im


def gen_image(size, x, y, flux, sky, rdnoise, model='gaussian', **kwargs):
    """Generate a full image of stars with noise."""
    im = gen_bkg(size, sky, rdnoise)

    if model == 'moffat':
        fwhm = kwargs.pop('fwhm')
        im += gen_stars_moffat(size, x, y, flux, fwhm)
    if model == 'gaussian':
        sigma = kwargs.pop('sigma', 2.0)
        theta = kwargs.pop('theta', 0)
        im += gen_stars_gaussian(size, x, y, flux, sigma, theta)

    # can pass the poisson noise
    if not kwargs.get('skip_poisson', False):
        # prevent negative number error
        negatives = np.where(im < 0)
        im = np.random.poisson(np.absolute(im))
        # restore the negatives
        im[negatives] = -im[negatives]
    return im
