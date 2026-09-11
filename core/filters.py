import numpy as np
import scipy.signal as sig
from joblib import Parallel, delayed
import warnings

'''
organize loading data and parse setttings file

then

csd - check 20, 40 or 60 steps
theta - gamma mod (als cds)
theta amp depth profile
theta phase

then

redo spike sorting using correct settings file
'''

def next_power_of_2(n):
    """
    Return the next power of 2 greater than or equal to 'n'.

    Parameters
    ----------
    n : int
        Input number.

    Returns
    -------
    next_power : int
        Next power of 2 greater than or equal to 'n'.

    Notes
    -----
    This function calculates the next power of 2 that is greater than or equal to the input 'n'.
    It finds the next power of 2 by performing bitwise operations, shifting bits to determine the nearest power of 2.
    """
    n -= 1                 # short for "n = n - 1"
    shift = 1
    while (n+1) & n:       # the operator "&" is a bitwise operator: it compares every bit of (n+1) and n, and returns those bits that are present in both
        n |= n >> shift    
        shift <<= 1        # the operator "<<" means "left bitwise shift": this comes down to "shift = shift**2"
    return n + 1



def hilbert(lfp):

    """
    Compute the Hilbert transform of the input signal(s).

    Parameters
    ----------
    lfp : numpy.ndarray
        Input signal. If 1D, treated as a single-channel signal. If 2D, each row represents a channel.

    Returns
    -------
    hilbert_result : numpy.ndarray
        Complex-valued array containing the Hilbert transform of the input signal(s). 
        If the input was 1D, the output is 1D. If the input was 2D, the output preserves the original shape.

    Raises
    ------
    ValueError
        If the input array is not 1D or 2D.

    Notes
    -----
    The Hilbert transform creates an analytic signal by adding a 90-degree phase-shifted version of the input signal.
    This function handles both single-channel and multi-channel signals, applying the transform independently to each channel.
    """
    
    if lfp.ndim == 2:
        n_channels, n_timepoints = lfp.shape
    elif lfp.ndim == 1:
        n_channels = 1
        n_timepoints = len(lfp)
        lfp = lfp.reshape((n_channels, n_timepoints))
    else:
        raise ValueError("Input array should be 1D or 2D")

    hilbert_result = np.zeros_like(lfp, dtype=np.complex128)  # Initialize complex-valued array

    for channel_idx in range(n_channels):
        channel_data = lfp[channel_idx]  # Extract data for each channel
        hilbert_channel = sig.hilbert(channel_data, next_power_of_2(len(channel_data)))  # Apply Hilbert transform
        hilbert_result[channel_idx] = hilbert_channel[range(len(channel_data))]  # Store the result for each channel

    return hilbert_result.squeeze() if n_channels == 1 else hilbert_result



def notch_filter(lfp, fs, notch_freq, quality_factor=30):
    """
    Apply a notch filter to remove specific frequency noise.
    
    Parameters
    ----------
    lfp : array-like
        Input signal (can be 1D or 2D with channels as rows)
    fs : float
        Sampling frequency in Hz
    notch_freq : float
        Frequency to be removed (e.g., 50 or 60 Hz for line noise)
    quality_factor : float, optional
        Quality factor defining the bandwidth (higher = narrower notch)
        Default is 30
    
    Returns
    -------
    filtered : array-like
        Notch-filtered signal with same shape as input
    
    Notes
    -----
    The quality factor determines the bandwidth: BW = notch_freq / quality_factor
    A higher quality factor results in a narrower notch filter.
    """
    # Validate inputs
    if notch_freq >= fs/2:
        raise ValueError(f"Notch frequency {notch_freq} Hz must be below Nyquist frequency {fs/2} Hz")
    
    if quality_factor <= 0:
        raise ValueError(f"Quality factor must be positive, got {quality_factor}")
    
    # Design and apply filter
    b_notch, a_notch = sig.iirnotch(notch_freq, quality_factor, fs)
    filtered = sig.filtfilt(b_notch, a_notch, lfp, axis=-1)
    
    return filtered

def bandpass_filter(lfp, fs, lowcut, highcut, order=4, notch_freq=None, quality_factor=30):
    """
    Apply bandpass, lowpass, highpass, and/or notch filtering to neural signals.
    
    Parameters
    ----------
    lfp : array-like
        Input signal. Can be 1D (single channel) or 2D (channels × timepoints)
    fs : float
        Sampling frequency in Hz
    lowcut : float
        Low cutoff frequency in Hz. Use 0 for lowpass filter
    highcut : float
        High cutoff frequency in Hz. Use 0 for highpass filter
    order : int, optional
        Order of the Butterworth filter (default is 3)
    notch_freq : float, optional
        Frequency for optional notch filtering (e.g., 50 or 60 Hz)
    quality_factor : float, optional
        Quality factor for notch filter (default is 30)
    
    Returns
    -------
    filtered : array-like
        Filtered signal with same shape as input
    
    Notes
    -----
    - filtfilt applies zero-phase filtering (forward-backward)
    - For bandpass: lowcut < highcut, both > 0
    - For lowpass: lowcut = 0, highcut > 0
    - For highpass: highcut = 0, lowcut > 0
    - Channels that are all NaN remain all NaN in the output
    """
    # Validate inputs
    if fs <= 0:
        raise ValueError(f"Sampling frequency must be positive, got {fs}")
    
    if lowcut < 0 or highcut < 0:
        raise ValueError("Cutoff frequencies cannot be negative")
    
    if lowcut >= fs/2 or highcut >= fs/2:
        raise ValueError(f"Cutoff frequencies must be below Nyquist frequency {fs/2} Hz")
    
    # Normalize frequencies
    nyq = 0.5 * fs
    low_norm = lowcut / nyq if lowcut > 0 else 0
    high_norm = highcut / nyq if highcut > 0 else 0
    
    # Make a copy
    filtered = lfp.copy()
    
    # Determine if input is 1D or 2D
    is_1d = False
    if filtered.ndim == 1:
        is_1d = True
        filtered = filtered.reshape(1, -1)
    
    # Apply notch filter if specified
    if notch_freq is not None:
        filtered = notch_filter(filtered, fs, notch_freq, quality_factor)
    
    # Apply main filter (bandpass, lowpass, or highpass)
    if low_norm == 0 and high_norm > 0:  # Lowpass
        if high_norm >= 1:
            raise ValueError(f"Normalized high cutoff {high_norm} must be < 1")
        sos = sig.butter(order, high_norm, btype='low', output='sos')
        
    elif high_norm == 0 and low_norm > 0:  # Highpass
        if low_norm >= 1:
            raise ValueError(f"Normalized low cutoff {low_norm} must be < 1")
        sos = sig.butter(order, low_norm, btype='high', output='sos')
        
    elif low_norm > 0 and high_norm > 0:  # Bandpass
        if low_norm >= high_norm:
            raise ValueError(f"Low cutoff {low_norm} must be less than high cutoff {high_norm}")
        if high_norm >= 1:
            raise ValueError(f"High cutoff {high_norm} must be < 1")
        sos = sig.butter(order, [low_norm, high_norm], btype='band', output='sos')
        
    else:
        raise ValueError(f"Invalid filter configuration: lowcut={lowcut}, highcut={highcut}")
    
    # Apply filter channel by channel, handling NaN channels
    for ch in range(filtered.shape[0]):
        # Check if entire channel is NaN
        if np.all(np.isnan(filtered[ch])):
            # Keep as all NaN (already is, just continue)
            continue
        
        # Filter the channel
        filtered[ch] = sig.sosfiltfilt(sos, filtered[ch], axis=-1)
    
    # Return to original shape if input was 1D
    if is_1d:
        filtered = filtered[0]
    
    return filtered







'''

def bandpass_filter(lfp, fs, lowcut, highcut, order=4, notch_freq=None, quality_factor=30):
    """
    Apply bandpass, lowpass, highpass, and/or notch filtering to neural signals.
    
    Parameters
    ----------
    lfp : array-like
        Input signal. Can be 1D (single channel) or 2D (channels × timepoints)
    fs : float
        Sampling frequency in Hz
    lowcut : float
        Low cutoff frequency in Hz. Use 0 for lowpass filter
    highcut : float
        High cutoff frequency in Hz. Use 0 for highpass filter
    order : int, optional
        Order of the Butterworth filter (default is 3)
    notch_freq : float, optional
        Frequency for optional notch filtering (e.g., 50 or 60 Hz)
    quality_factor : float, optional
        Quality factor for notch filter (default is 30)
    
    Returns
    -------
    filtered : array-like
        Filtered signal with same shape as input
    
    Notes
    -----
    - filtfilt applies zero-phase filtering (forward-backward)
    - For bandpass: lowcut < highcut, both > 0
    - For lowpass: lowcut = 0, highcut > 0
    - For highpass: highcut = 0, lowcut > 0
    """
    # Validate inputs
    if fs <= 0:
        raise ValueError(f"Sampling frequency must be positive, got {fs}")
    
    if lowcut < 0 or highcut < 0:
        raise ValueError("Cutoff frequencies cannot be negative")
    
    if lowcut >= fs/2 or highcut >= fs/2:
        raise ValueError(f"Cutoff frequencies must be below Nyquist frequency {fs/2} Hz")
    
    # Normalize frequencies
    nyq = 0.5 * fs
    low_norm = lowcut / nyq if lowcut > 0 else 0
    high_norm = highcut / nyq if highcut > 0 else 0
    
    # Apply filters sequentially
    filtered = lfp.copy()
    
    # Apply notch filter if specified
    if notch_freq is not None:
        filtered = notch_filter(filtered, fs, notch_freq, quality_factor)
    
    # Apply main filter (bandpass, lowpass, or highpass)
    if low_norm == 0 and high_norm > 0:  # Lowpass
        if high_norm >= 1:
            raise ValueError(f"Normalized high cutoff {high_norm} must be < 1")
        sos = sig.butter(order, high_norm, btype='low', output='sos')
        
    elif high_norm == 0 and low_norm > 0:  # Highpass
        if low_norm >= 1:
            raise ValueError(f"Normalized low cutoff {low_norm} must be < 1")
        sos = sig.butter(order, low_norm, btype='high', output='sos')
        
    elif low_norm > 0 and high_norm > 0:  # Bandpass
        if low_norm >= high_norm:
            raise ValueError(f"Low cutoff {low_norm} must be less than high cutoff {high_norm}")
        if high_norm >= 1:
            raise ValueError(f"High cutoff {high_norm} must be < 1")
        sos = sig.butter(order, [low_norm, high_norm], btype='band', output='sos')
        
    else:
        raise ValueError(f"Invalid filter configuration: lowcut={lowcut}, highcut={highcut}")
    
    # Apply filter using sosfiltfilt for better numerical stability
    filtered = sig.sosfiltfilt(sos, filtered, axis=-1)
    
    return filtered
'''

def calculate_amplitude_bands(amp_center_freqs, phase_low, phase_high, fs):
    """
    Calculate frequency band bounds for amplitude frequencies based on phase frequency range.
    
    Parameters
    ----------
    amp_center_freqs : array-like
        Center frequencies for amplitude bands (e.g., [20, 25, 65])
    phase_low : float
        Low cutoff frequency of phase signal (e.g., 4 for theta)
    phase_high : float
        High cutoff frequency of phase signal (e.g., 10 for theta)
    fs : float
        Sampling frequency in Hz
    
    Returns
    -------
    freq_ranges : list of tuples
        List of (low_bound, high_bound) tuples for each amplitude center frequency
    
    Notes
    -----
    Bounds are calculated as:
        low_bound = amp_center - center_phase_freq - 1
        high_bound = amp_center + center_phase_freq + 1
    where center_phase_freq = (phase_low + phase_high) / 2
    
    Bounds are clipped to [0, nyquist]
    """
    nyquist = fs / 2
    center_phase_freq = (phase_low + phase_high) / 2
    
    freq_ranges = []
    for amp_center in amp_center_freqs:
        # Calculate bounds
        low_bound = amp_center - center_phase_freq - 1
        high_bound = amp_center + center_phase_freq + 1
        
        # Clip to valid frequency range
        low_bound = max(0, low_bound)
        high_bound = min(nyquist, high_bound)
        
        # Ensure low_bound < high_bound
        if low_bound >= high_bound:
            warnings.warn(f"Frequency band for {amp_center} Hz resulted in invalid bounds "
                         f"({low_bound:.1f}-{high_bound:.1f} Hz). Adjusting...")
            # If bounds are invalid, use a small band around the center
            low_bound = max(0, amp_center - 2)
            high_bound = min(nyquist, amp_center + 2)
        
        freq_ranges.append((low_bound, high_bound))
    
    return freq_ranges


def apply_filtering_parallel(lfp, srate, freq_ranges, filter_order=3, notch_freq=None, 
                            quality_factor=30, n_jobs=-1, verbose=0):
    """
    Apply bandpass filtering in parallel across multiple frequency ranges.
    
    Parameters
    ----------
    lfp : array-like
        Input signal. Can be 1D (single channel) or 2D (channels × timepoints)
    srate : float
        Sampling rate in Hz
    freq_ranges : list of tuples
        List of (low_freq, high_freq) tuples defining frequency bands to extract
    filter_order : int, optional
        Order of Butterworth filter (default is 3)
    notch_freq : float, optional
        Frequency for optional notch filtering
    quality_factor : float, optional
        Quality factor for notch filter (default is 30)
    n_jobs : int, optional
        Number of parallel jobs. -1 uses all available cores (default is -1)
    verbose : int, optional
        Verbosity level for parallel processing (default is 0)
    
    Returns
    -------
    amplitudes : ndarray
        3D array of shape (n_freq_ranges, n_channels, n_timepoints) with
        instantaneous amplitudes from Hilbert transform
    """
    
    if not freq_ranges:
        raise ValueError("freq_ranges must contain at least one frequency band")
    
    # Validate frequency ranges
    nyquist = srate / 2
    for i, (low, high) in enumerate(freq_ranges):
        if low < 0 or high > nyquist:
            raise ValueError(f"Frequency band {i}: ({low}, {high}) must be within [0, {nyquist}] Hz")
        if low >= high:
            raise ValueError(f"Frequency band {i}: low bound ({low}) must be less than high bound ({high})")
    
    # Process each frequency range in parallel
    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_process_frequency_band)(
            lfp, srate, low_freq, high_freq, filter_order, notch_freq, quality_factor
        )
        for low_freq, high_freq in freq_ranges
    )
    
    # Stack results
    return np.array(results)


def _process_frequency_band(lfp, srate, low_freq, high_freq, order, notch_freq, quality_factor):
    """
    Helper function for parallel processing of a single frequency band.
    """
    filtered = bandpass_filter(lfp, srate, low_freq, high_freq, 
                       order=order, notch_freq=notch_freq, 
                       quality_factor=quality_factor)
    
    # Compute Hilbert transform and extract amplitude
    analytic_signal = hilbert(filtered)
    amplitude = np.abs(analytic_signal)
    
    return amplitude


def chebyshev_antialias_decimate(lfp, fs, target_fs, cheby_order=8, max_ripple_db=0.1,
                                  decimate_order=8):
    """
    Anti-aliasing filter followed by decimation, replicating the method described in:

        "LFP signals underwent initial filtering using an 8th-order Chebyshev type I
        anti-aliasing filter, applied to the wide-band signals sampled at 20 kHz.
        These filtered signals were then downsampled to a rate of 1,250 Hz, employing
        the decimate function within Scipy's signal submodule."

    The pipeline is:
        1. Apply a zero-phase 8th-order Chebyshev Type I lowpass filter with a
           cutoff just below the new Nyquist (target_fs / 2), attenuating aliasing
           content before downsampling.
        2. Decimate using ``scipy.signal.decimate``, which internally applies a
           further IIR lowpass anti-aliasing filter and then downsamples by an
           integer factor.

    When the total decimation factor is large (e.g., 20000 → 1250 = factor 16),
    ``decimate`` is applied in stages to avoid numerical instability.

    Parameters
    ----------
    lfp : array-like
        Input signal. Can be 1D (single channel) or 2D (channels × timepoints).
        Data should be in its original wide-band form (not pre-filtered).
    fs : float
        Sampling frequency of the input signal in Hz (e.g., 20000).
    target_fs : float
        Desired output sampling frequency in Hz (e.g., 1250).
    cheby_order : int, optional
        Order of the Chebyshev Type I anti-aliasing filter (default is 8).
    max_ripple_db : float, optional
        Maximum passband ripple in dB for the Chebyshev filter (default is 0.1).
        Lower values → flatter passband but slower rolloff.
    decimate_order : int, optional
        Order of the IIR filter used internally by ``scipy.signal.decimate``
        (default is 8).

    Returns
    -------
    decimated : ndarray
        Downsampled signal with shape matching input but with
        ``ceil(n_samples / decimation_factor)`` timepoints.

    Notes
    -----
    **Why Chebyshev Type I?**
        Compared to a Butterworth filter of the same order, a Chebyshev Type I
        filter has a steeper transition band at the cost of a small equiripple in
        the passband. For anti-aliasing before decimation the steeper cutoff is
        desirable; the passband ripple (0.1 dB by default) is negligible for
        typical LFP analyses.

    **Why ``decimate`` rather than ``resample_poly``?**
        ``scipy.signal.decimate`` is the standard tool for integer-factor
        downsampling in electrophysiology pipelines. It applies its own lowpass
        IIR filter before subsampling, providing a second layer of anti-aliasing.
        ``resample_poly`` (polyphase FIR) is excellent for non-integer ratios but
        does not add an explicit anti-aliasing stage beyond its own FIR kernel.

    **Two-stage decimation**
        Large decimation factors (> ~13) can make a single ``decimate`` call
        numerically unstable. This function automatically factors the total
        decimation ratio into two stages when needed.

    Raises
    ------
    ValueError
        If ``target_fs`` >= ``fs`` or if the decimation factor is not an integer.

    Examples
    --------
    >>> # Replicate the paper's preprocessing on Neuropixels AP-band data
    >>> decimated = chebyshev_antialias_decimate(raw_data, fs=20000, target_fs=1250)
    """
    if target_fs >= fs:
        raise ValueError(f"target_fs ({target_fs}) must be less than fs ({fs})")

    decimation_factor = fs / target_fs
    if not np.isclose(decimation_factor, round(decimation_factor)):
        raise ValueError(
            f"Decimation factor fs/target_fs = {decimation_factor:.4f} must be an integer. "
            f"Got fs={fs}, target_fs={target_fs}."
        )
    decimation_factor = int(round(decimation_factor))

    # ── Step 1: Chebyshev Type I anti-aliasing lowpass filter ─────────────
    # Cutoff at the new Nyquist (target_fs / 2), normalised by current Nyquist
    nyquist = fs / 2.0
    cutoff_norm = (target_fs / 2.0) / nyquist  # value in (0, 1)

    b_cheby, a_cheby = sig.cheby1(cheby_order, max_ripple_db, cutoff_norm, btype='low')
    filtered = sig.filtfilt(b_cheby, a_cheby, lfp, axis=-1)

    # ── Step 2: Decimate (with its own internal anti-aliasing filter) ──────
    # For large factors, split into two stages to keep numerical stability.
    MAX_SINGLE_FACTOR = 13  # rule-of-thumb threshold

    if decimation_factor <= MAX_SINGLE_FACTOR:
        decimated = sig.decimate(filtered, decimation_factor, n=decimate_order,
                                 ftype='iir', zero_phase=True, axis=-1)
    else:
        # Factor the decimation into two integer stages, as close to sqrt as possible
        factor1 = int(np.floor(np.sqrt(decimation_factor)))
        while decimation_factor % factor1 != 0:
            factor1 -= 1
        factor2 = decimation_factor // factor1

        stage1 = sig.decimate(filtered, factor1, n=decimate_order,
                              ftype='iir', zero_phase=True, axis=-1)
        decimated = sig.decimate(stage1, factor2, n=decimate_order,
                                 ftype='iir', zero_phase=True, axis=-1)

    return decimated


def preprocess_raw_signal(lfp, fs, target_fs,
                           method='decimate',
                           low_cut=None, high_cut=None,
                           filter_order=4,
                           notch_freq=None, quality_factor=30,
                           cheby_order=8, max_ripple_db=0.1,
                           decimate_order=8):
    """
    Unified entry point for anti-aliasing and downsampling of wide-band LFP.

    Two methods are available:

    ``'decimate'`` (default)
        Replicates the published preprocessing pipeline:
        8th-order Chebyshev Type I lowpass anti-aliasing filter followed by
        ``scipy.signal.decimate``.  See ``chebyshev_antialias_decimate`` for
        full details.  Recommended when working with AP-band data at 20 kHz
        and targeting standard LFP rates (e.g., 1250 Hz).

    ``'resample_poly'``
        The original approach: optional bandpass (Butterworth) + optional notch
        filtering, then polyphase resampling via ``scipy.signal.resample_poly``.
        Better suited to non-integer decimation ratios or when you need precise
        control over the passband with a flat Butterworth response.

    Parameters
    ----------
    lfp : array-like
        Input signal. 1D (single channel) or 2D (channels × timepoints).
    fs : float
        Input sampling frequency in Hz.
    target_fs : float
        Desired output sampling frequency in Hz.
    method : {'decimate', 'resample_poly'}, optional
        Preprocessing pipeline to use (default is ``'decimate'``).
    low_cut : float, optional
        Low cutoff for bandpass filter in Hz.  Used only with
        ``method='resample_poly'``.  Pass 0 for lowpass-only.
    high_cut : float, optional
        High cutoff for bandpass filter in Hz.  Used only with
        ``method='resample_poly'``.  Pass 0 for highpass-only.
    filter_order : int, optional
        Butterworth filter order for ``method='resample_poly'`` (default 4).
    notch_freq : float, optional
        Line-noise frequency to notch out in Hz (both methods).
    quality_factor : float, optional
        Quality factor for the notch filter (default 30).
    cheby_order : int, optional
        Chebyshev filter order for ``method='decimate'`` (default 8).
    max_ripple_db : float, optional
        Chebyshev passband ripple in dB for ``method='decimate'`` (default 0.1).
    decimate_order : int, optional
        IIR order used by ``scipy.signal.decimate`` for ``method='decimate'``
        (default 8).

    Returns
    -------
    downsampled : ndarray
        Processed signal with the same number of channels but fewer timepoints.

    Examples
    --------
    >>> # Paper method (default) – 20 kHz → 1250 Hz
    >>> lfp_ds = preprocess_raw_signal(raw_data, fs=20000, target_fs=1250)

    >>> # Original resample_poly method with theta bandpass
    >>> lfp_ds = preprocess_raw_signal(raw_data, fs=20000, target_fs=1250,
    ...                                method='resample_poly',
    ...                                low_cut=4, high_cut=10)

    >>> # Paper method + notch filter for 50 Hz line noise
    >>> lfp_ds = preprocess_raw_signal(raw_data, fs=20000, target_fs=1250,
    ...                                notch_freq=50)
    """
    if method not in ('decimate', 'resample_poly'):
        raise ValueError(f"Unknown method '{method}'. Choose 'decimate' or 'resample_poly'.")

    # Optional notch filter (applied before any downsampling, same for both methods)
    if notch_freq is not None:
        lfp = notch_filter(lfp, fs, notch_freq, quality_factor)

    if method == 'decimate':
        return chebyshev_antialias_decimate(
            lfp, fs, target_fs,
            cheby_order=cheby_order,
            max_ripple_db=max_ripple_db,
            decimate_order=decimate_order
        )

    elif method == 'resample_poly':
        # Optional bandpass filter before resampling
        if low_cut is not None and high_cut is not None:
            lfp = bandpass_filter(lfp, fs, low_cut, high_cut,
                                  order=filter_order)

        # Polyphase resampling (handles non-integer ratios cleanly)
        decimation_factor = int(np.round(fs / target_fs))
        return sig.resample_poly(lfp, up=1, down=decimation_factor, axis=-1)

    else:
        raise ValueError(f"Invalid method: '{method}'. Expected 'decimate' or 'resample_poly'.")