import numpy as np


#####   Functions   #####
# Logbinning
def resampling_matrix_nonuniform(lorig, lresam, extrap = False):
    '''
    Logbinning stolen from some astro people: https://pypi.org/project/PySTARLIGHT/

    Compute resampling matrix R_o2r, useful to convert a spectrum sampled at
    wavelengths lorig to a new grid lresamp. Here, there is no necessity to have constant gris as on :py:func:`ReSamplingMatrix`.
    Input arrays lorig and lresamp are the bin centres of the original and final lambda-grids.
    ResampMat is a Nlresamp x Nlorig matrix, which applied to a vector F_o (with Nlorig entries) returns
    a Nlresamp elements long vector F_r (the resampled spectrum):

        [[ResampMat]] [F_o] = [F_r]

    Warning! lorig and lresam MUST be on ascending order!


    Parameters
    ----------
    lorig : array_like
            Original spectrum lambda array.

    lresam : array_like
             Spectrum lambda array in which the spectrum should be sampled.

    extrap : boolean, optional
           Extrapolate values, i.e., values for lresam < lorig[0]  are set to match lorig[0] and
                                     values for lresam > lorig[-1] are set to match lorig[-1].


    Returns
    -------
    ResampMat : array_like
                Resample matrix.

    Examples
    --------
    >>> lorig = np.linspace(3400, 8900, 9000) * 1.001
    >>> lresam = np.linspace(3400, 8900, 5000)
    >>> forig = np.random.normal(size=len(lorig))**2
    >>> matrix = slut.resampling_matrix_nonuniform(lorig, lresam)
    >>> fresam = np.dot(matrix, forig)
    >>> print np.trapz(forig, lorig), np.trapz(fresam, lresam)
    '''

    # Init ResampMatrix
    matrix = np.zeros((len(lresam), len(lorig)))

    # Define lambda ranges (low, upp) for original and resampled.
    lo_low = np.zeros(len(lorig))
    lo_low[1:] = (lorig[1:] + lorig[:-1])/2
    lo_low[0] = lorig[0] - (lorig[1] - lorig[0])/2

    lo_upp = np.zeros(len(lorig))
    lo_upp[:-1] = lo_low[1:]
    lo_upp[-1] = lorig[-1] + (lorig[-1] - lorig[-2])/2

    lr_low = np.zeros(len(lresam))
    lr_low[1:] = (lresam[1:] + lresam[:-1])/2
    lr_low[0] = lresam[0] - (lresam[1] - lresam[0])/2

    lr_upp = np.zeros(len(lresam))
    lr_upp[:-1] = lr_low[1:]
    lr_upp[-1] = lresam[-1] + (lresam[-1] - lresam[-2])/2


    # Iterate over resampled lresam vector
    for i_r in range(len(lresam)):

        # Find in which bins lresam bin within lorig bin
        bins_resam = np.where( (lr_low[i_r] < lo_upp) & (lr_upp[i_r] > lo_low) )[0]

        # On these bins, eval fraction of resamled bin is within original bin.
        for i_o in bins_resam:

            aux = 0

            d_lr = lr_upp[i_r] - lr_low[i_r]
            d_lo = lo_upp[i_o] - lo_low[i_o]
            d_ir = lo_upp[i_o] - lr_low[i_r]  # common section on the right
            d_il = lr_upp[i_r] - lo_low[i_o]  # common section on the left

            # Case 1: resampling window is smaller than or equal to the original window.
            # This is where the bug was: if an original bin is all inside the resampled bin, then
            # all flux should go into it, not then d_lr/d_lo fraction. --Natalia@IoA - 21/12/2012
            if (lr_low[i_r] > lo_low[i_o]) & (lr_upp[i_r] < lo_upp[i_o]):
                aux += 1.

            # Case 2: resampling window is larger than the original window.
            if (lr_low[i_r] < lo_low[i_o]) & (lr_upp[i_r] > lo_upp[i_o]):
                aux += d_lo / d_lr

            # Case 3: resampling window is on the right of the original window.
            if (lr_low[i_r] > lo_low[i_o]) & (lr_upp[i_r] > lo_upp[i_o]):
                aux += d_ir / d_lr

            # Case 4: resampling window is on the left of the original window.
            if (lr_low[i_r] < lo_low[i_o]) & (lr_upp[i_r] < lo_upp[i_o]):
                aux += d_il / d_lr

            matrix[i_r, i_o] += aux


    # Fix matrix to be exactly = 1 ==> TO THINK
    #print np.sum(matrix), np.sum(lo_upp - lo_low), (lr_upp - lr_low).shape


    # Fix extremes: extrapolate if needed
    if (extrap):

        bins_extrapl = np.where( (lr_low < lo_low[0])  )[0]
        bins_extrapr = np.where( (lr_upp > lo_upp[-1]) )[0]

        if (len(bins_extrapl) > 0) & (len(bins_extrapr) > 0):
            io_extrapl = np.where( (lo_low >= lr_low[bins_extrapl[0]])  )[0][0]
            io_extrapr = np.where( (lo_upp <= lr_upp[bins_extrapr[0]])  )[0][-1]

            matrix[bins_extrapl, io_extrapl] = 1.
            matrix[bins_extrapr, io_extrapr] = 1.


    return matrix

def logbin_ASD(log_ff, linear_ff, linear_ASD):
    '''Logbins an ASD given some log spaced frequency vector.
    Inputs:
    log_ff is the final vector we want the ASD to be spaced at
    linear_ff is the original frequency vector
    linear_ASD is the ASD
    '''
    matrix = resampling_matrix_nonuniform(linear_ff, log_ff)
    log_ASD = np.dot(matrix, linear_ASD)
    return log_ASD

def linear_log_ASD(log_ff, linear_ff, linear_ASD):
    '''
    Creates a linear- and log-binned ASD vector from overlapping linear and log frequency vectors,
    such that the coarsest frequency vector is used.
    This avoids the problem of logbinning where the low frequency points have too
    much resolution, i.e. the FFT binwidth > log_ff[1] - log_ff[0].

    Inputs:
    linear_ff  = linear frequency vector. Will be used for low frequency points.
    log_ff     = log frequency vector.  Will be used for high frequency points.
    linear_ASD = linear ASD. Should be the same length as linear_ff, usual output of scipy.signal.welch().

    Outputs:
    fff = stitched frequency vector of linear and log points
    linlog_ASD = stitched ASD of linear and log points
    '''
    df = linear_ff[1] - linear_ff[0]
    dfflog = np.diff(log_ff)
    log_index = np.argwhere(dfflog > df)[0][0] # first point where fflog has less resolution than the normal freq vector
    cutoff = log_ff[log_index]                 # cutoff frequency
    high_fff = log_ff[log_index+1:]

    linear_index = np.argwhere(cutoff < linear_ff)[0][0] # find where the cutoff frequency is first less than the linear frequency vector
    low_fff = linear_ff[:linear_index]
    low_ASD = linear_ASD[:linear_index]

    fff = np.concatenate((low_fff, high_fff)) # make the full frequency vector

    matrix = resampling_matrix_nonuniform(linear_ff, high_fff)
    high_ASD = np.dot(matrix, linear_ASD) # get HF part of spectrum

    linlog_ASD = np.concatenate((low_ASD, high_ASD))

    return fff, linlog_ASD

