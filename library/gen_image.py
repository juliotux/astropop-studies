import numpy as np
from astropy.utils import NumpyRNGContext
from astropop.math.models import MoffatEquations, GaussianEquations
from astropop.py_utils import check_number


def gen_bkg(size, level, rdnoise, rng_seed=123, dtype='f8'):
    """Generate a simple background image."""
    # create a level image
    im = np.ones(size, dtype)*level

    # reate the gaussian read noise image to sum
    if rdnoise > 0:
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
        im += MoffatEquations.model_2d(grid_x, grid_y, xi, yi,
                                       width=alpha, power=beta, flux=fi, sky=0)

    return im


def gen_stars_gaussian(size, x, y, flux, sigma, theta, truncate=None):
    """Render Gaussians, optionally restricting each star to a local stamp.

    ``truncate`` is the bounding radius in Gaussian standard deviations. None
    retains full-image evaluation. Eight omits tails below exp(-32) of the peak;
    rotated elliptical profiles use their projected widths for the bounds.
    """
    im = np.zeros(size)
    if truncate is None:
        grid_y, grid_x = np.indices(size)
    elif not np.isfinite(truncate) or truncate <= 0:
        raise ValueError('truncate must be finite and positive')

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
        if truncate is None:
            target = im
        else:
            # Project the rotated Gaussian covariance onto the detector axes.
            angle = np.radians(ti)
            rx = truncate*np.hypot(sxi*np.cos(angle), syi*np.sin(angle))
            ry = truncate*np.hypot(sxi*np.sin(angle), syi*np.cos(angle))
            left = max(0, int(np.ceil(xi-rx)))
            right = min(size[1], int(np.floor(xi+rx))+1)
            bottom = max(0, int(np.ceil(yi-ry)))
            top = min(size[0], int(np.floor(yi+ry))+1)
            if left >= right or bottom >= top:
                continue
            grid_y, grid_x = np.ogrid[bottom:top, left:right]
            target = im[bottom:top, left:right]
        target += GaussianEquations.model_2d(grid_x, grid_y, xi, yi, flux=fi,
                                              sigma_x=sxi, sigma_y=syi,
                                              theta=ti, sky=0)

    return im


def gen_image(size, x, y, flux, sky, rdnoise, model='gaussian', **kwargs):
    """Generate a full image of stars with noise.

    For Gaussian stars, ``truncate`` optionally bounds the rendering stamps in
    standard deviations; the default None evaluates the complete image.
    """
    im = gen_bkg(size, sky, rdnoise)

    if model == 'moffat':
        fwhm = kwargs.pop('fwhm')
        im += gen_stars_moffat(size, x, y, flux, fwhm)
    if model == 'gaussian':
        sigma = kwargs.pop('sigma', 2.0)
        theta = kwargs.pop('theta', 0)
        im += gen_stars_gaussian(size, x, y, flux, sigma, theta,
                                 truncate=kwargs.pop('truncate', None))

    # can pass the poisson noise
    if not kwargs.get('skip_poisson', False):
        # prevent negative number error
        negatives = np.where(im < 0)
        im = np.random.poisson(np.absolute(im))
        # restore the negatives
        im[negatives] = -im[negatives]
    return im
