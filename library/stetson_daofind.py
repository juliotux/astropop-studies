# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.table import Table
from scipy.ndimage.filters import convolve
from astropop.logger import logger
import warnings


def symmetry_roundness(chunk, nhalf):
    """Compute the folding roundness."""
    # Quads
    # 3 3 4 4 4
    # 3 3 4 4 4
    # 3 3 x 1 1
    # 2 2 2 1 1
    # 2 2 2 1 1
    chunk = np.array(chunk)
    chunk[nhalf, nhalf] = 0  # copy and setcentral pixel to 0

    q1 = chunk[nhalf:, nhalf+1:].sum()
    q2 = chunk[nhalf+1:, :nhalf+1].sum()
    q3 = chunk[:nhalf+1, :nhalf].sum()
    q4 = chunk[:nhalf, nhalf:].sum()

    sum2 = -q1+q2-q3+q4
    sum4 = q1+q2+q3+q4

    # ignore divide-by-zero RuntimeWarning
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        sround = 2.0 * sum2 / sum4
    return sround


def stetson_find_peaks(h, hmin, mask, pixels, middle, n_x, n_y):
    """Stetson way to find peaks."""
    mask = np.array(mask)  # work with a copy
    mask[middle, middle] = 0  # From now on we exclude the central pixel
    pixels = np.sum(mask)

    good = np.where(mask)  # "good" identifies position of valid pixels
    xx = good[1] - middle  # x and y coordinate of valid pixels
    yy = good[0] - middle  # relative to the center
    index = np.where(h >= hmin)  # Valid image pixels are greater than hmin
    nfound = len(index)
    logger.debug('%i pixels above threshold', nfound)

    if nfound == 0:  # Any maxima found?
        logger.warning('No maxima exceed input threshold of %f', hmin)
        return

    for i in range(pixels):
        hy = index[0]+yy[i]
        hx = index[1]+xx[i]
        hgood = np.where((hy < n_y) & (hx < n_x) & (hy >= 0) & (hx >= 0))[0]
        stars = np.where(np.greater_equal(h[index[0][hgood], index[1][hgood]],
                                          h[hy[hgood], hx[hgood]]))
        nfound = len(stars)
        if nfound == 0:  # Do valid local maxima exist?
            logger.warning('No maxima exceed input threshold of %f', hmin)
            return
        index = np.array([index[0][hgood][stars], index[1][hgood][stars]])

    ix = index[1]  # X index of local maxima
    iy = index[0]  # Y index of local maxima
    ngood = len(index[0])
    logger.debug('%i local maxima located above threshold', ngood)
    return ix, iy, ngood


def stetson_image_params(fwhm, snr, noise, data, background):
    """General Stetson kernel and image params."""
    hmin = np.median(snr*noise)

    image = data.astype(np.float64) - background
    maxbox = 13  # Maximum size of convolution box in pixels

    # Get information about the input image
    type = np.shape(image)
    if len(type) != 2:
        raise ValueError('data array must be 2 dimensional')
    n_x = type[1]
    n_y = type[0]
    logger.debug('Input Image Size is %ix%i', n_x, n_y)

    if fwhm < 0.5:
        raise ValueError('Supplied FWHM must be at least 0.5 pixels')

    radius = np.max([0.637*fwhm, 2.001])
    nhalf = np.min([int(radius), int((maxbox-1)/2.)])
    nbox = 2*nhalf + 1  # number of pixels in side of convolution box
    middle = nhalf  # Index of central pixel

    sigsq = (fwhm*gaussian_fwhm_to_sigma)**2

    return image, hmin, n_x, n_y, radius, nhalf, nbox, middle, sigsq


def stetson_source_chunk(image, ix, iy, nhalf):
    """Extract a source image chunk."""
    return image[iy-nhalf:iy+nhalf+1,
                 ix-nhalf:ix+nhalf+1]


def stetson_sharpness(temp, middle, mask, d):
    """Stetson compute of sharpness."""
    mask = np.array(mask)  # work with a copy
    mask[middle, middle] = 0
    sharp = temp[middle, middle] - (np.sum(mask*temp))/np.sum(mask)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        sharp /= d
    return sharp


def stetson_roundness(temp, c1):
    """Stetson compute of roundness."""
    dx = np.sum(np.sum(temp, axis=0)*c1)
    dy = np.sum(np.sum(temp, axis=1)*c1)
    if (dx <= 0) or (dy <= 0):
        return np.nan
    return 2*(dx-dy) / (dx + dy)


def stetson_kernels(radius, nhalf, nbox, sigsq):
    """Stetson kernel generation."""
    # Mask identifies valid pixels in convolution box
    mask = np.zeros([nbox, nbox], dtype='int8')
    g = np.zeros([nbox, nbox])  # Gaussian convolution kernel

    row2 = (np.arange(nbox)-nhalf)**2
    for i in range(nhalf+1):
        temp = row2 + i**2
        g[nhalf-i, :] = temp
        g[nhalf+i, :] = temp

    g_row = np.where(g <= radius**2)
    # MASK is complementary to SKIP in Stetson's Fortran
    mask[g_row[0], g_row[1]] = 1
    good = np.where(mask)  # Value of c are now equal to distance to center
    pixels = len(good[0])

    g = np.exp(-0.5*g/sigsq)

    c = g*mask  # Convolution kernel now in c
    sumc = np.sum(c)
    sumcsq = np.sum(c**2) - sumc**2/pixels
    sumc = sumc/pixels
    c[good[0], good[1]] = (c[good[0], good[1]] - sumc)/sumcsq
    c1 = np.exp(-.5*row2/sigsq)
    sumc1 = np.sum(c1)/nbox
    sumc1sq = np.sum(c1**2) - sumc1
    c1 = (c1-sumc1)/sumc1sq

    return mask, g, pixels, c, c1


def daofind_stetson(data, snr, background, noise, fwhm):
    """Find sources using DAOfind algorithm.

    For testing purpouse only.

    Translated from IDL Astro package by D. Jones. Original function available
    at PythonPhot package. https://github.com/djones1040/PythonPhot
    The function recieved some improvements to work better.
    The function was also modified to keep computing statistics, even with bad
    roundness and sharpness.
    """
    # Compute hmin based on snr, background and noise
    params = stetson_image_params(fwhm, snr, noise, data, background)
    image, hmin, n_x, n_y, radius, nhalf, nbox, middle, sigsq = params
    mask, g, pixels, c, c1 = stetson_kernels(radius, nhalf, nbox, sigsq)

    # Compute quantities for centroid computations that can be used for all
    # stars
    xwt = np.zeros([nbox, nbox])
    wt = nhalf - np.abs(np.arange(nbox)-nhalf) + 1
    for i in range(nbox):
        xwt[i, :] = wt
    ywt = np.transpose(xwt)
    sgx = np.sum(g*xwt, 1)
    p = np.sum(wt)
    sgy = np.sum(g*ywt, 0)
    sumgx = np.sum(wt*sgy)
    sumgy = np.sum(wt*sgx)
    sumgsqy = np.sum(wt*sgy*sgy)
    sumgsqx = np.sum(wt*sgx*sgx)
    vec = nhalf - np.arange(nbox)
    dgdx = sgy*vec
    dgdy = sgx*vec
    sdgdxs = np.sum(wt*dgdx**2)
    sdgdx = np.sum(wt*dgdx)
    sdgdys = np.sum(wt*dgdy**2)
    sdgdy = np.sum(wt*dgdy)
    sgdgdx = np.sum(wt*sgy*dgdx)
    sgdgdy = np.sum(wt*sgx*dgdy)

    h = convolve(image, c)  # Convolve image with kernel "c"

    minh = np.min(h)
    h[:, 0:nhalf] = minh
    h[:, n_x-nhalf:n_x] = minh
    h[0:nhalf, :] = minh
    h[n_y - nhalf: n_y - 1, :] = minh

    ix, iy, ngood = stetson_find_peaks(h, hmin, mask, pixels, middle, n_x, n_y)

    x = np.full(ngood, fill_value=np.nan)
    y = np.full(ngood, fill_value=np.nan)
    flux = np.full(ngood, fill_value=np.nan)
    sharp = np.full(ngood, fill_value=np.nan)
    roundness = np.full(ngood, fill_value=np.nan)

    #  Loop over star positions and compute statistics
    for i in range(ngood):
        temp = stetson_source_chunk(image, ix[i], iy[i], nhalf)
        d = h[iy[i], ix[i]]  # "d" is actual pixel intensity
        nstar = i

        #  Compute Sharpness statistic
        sharp[nstar] = stetson_sharpness(temp, middle, mask, d)

        #   Compute Roundness statistic
        roundness[nstar] = stetson_roundness(temp, c1)

        # Centroid computation: The centroid computation was modified in
        # Mar 2008 and now differs from DAOPHOT which multiplies the
        # correction dx by 1/(1+abs(dx)). The DAOPHOT method is more robust
        # (e.g. two different sources will not merge) especially in a package
        # where the centroid will be subsequently be redetermined using PSF
        # fitting. However, it is less accurate, and introduces biases in the
        # centroid histogram. The change here is the same made in the
        # IRAF DAOFIND routine
        # (see http://iraf.net/article.php?story=7211;query=daofind )
        sd = np.sum(temp*ywt, axis=0)
        sumgd = np.sum(wt*sgy*sd)
        sumd = np.sum(wt*sd)
        sddgdx = np.sum(wt*sd*dgdx)
        hx = (sumgd - sumgx*sumd/p) / (sumgsqy - sumgx**2/p)

        # HX is the height of the best-fitting marginal Gaussian. If this is
        # not positive then the centroid does not make sense
        if (hx <= 0):
            continue
        skylvl = (sumd - hx*sumgx)/p
        dx = (sgdgdx - (sddgdx-sdgdx*(hx*sumgx + skylvl*p)))/(hx*sdgdxs/sigsq)
        if np.abs(dx) >= nhalf:
            continue
        x[nstar] = ix[i] + dx  # X centroid in original array
        # Find Y centroid
        sd = np.sum(temp*xwt, axis=1)
        sumgd = np.sum(wt*sgx*sd)
        sumd = np.sum(wt*sd)
        sddgdy = np.sum(wt*sd*dgdy)
        hy = (sumgd - sumgy*sumd/p) / (sumgsqx - sumgy**2/p)
        if (hy <= 0):
            continue

        skylvl = (sumd - hy*sumgy)/p
        dy = (sgdgdy - (sddgdy-sdgdy*(hy*sumgy + skylvl*p)))/(hy*sdgdys/sigsq)
        if np.abs(dy) >= nhalf:
            continue
        y[nstar] = iy[i] + dy  # Y centroid in original array
        flux[nstar] = d

    t = Table([x, y, flux, sharp, roundness],
              names=('x', 'y', 'flux', 'sharp', 'round'))

    return t
