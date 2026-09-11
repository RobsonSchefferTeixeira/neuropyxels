"""
Wavelet Transform Analysis for Neural Signals

This module provides functions for time-frequency analysis using Morlet wavelets,
optimized for analyzing Local Field Potential (LFP) and other neural signals.

Features:
- Parallel processing for efficient computation
- Flexible output: complex signal, amplitude, phase, or power
- Adaptive cycle selection across frequency ranges
- Multiple normalization methods
- Edge artifact handling

Example:
    >>> import numpy as np
    >>> from wavelet_transform import compute_wavelet_transform
    >>> 
    >>> # Generate sample signal
    >>> srate = 1000  # Hz
    >>> time = np.linspace(0, 2, 2000)
    >>> signal = np.sin(2 * np.pi * 10 * time) + 0.5 * np.sin(2 * np.pi * 30 * time)
    >>> 
    >>> # Compute wavelet transform
    >>> freqs = np.linspace(5, 50, 20)
    >>> result = compute_wavelet_transform(signal, srate, freqs, output='all')
    >>> 
    >>> # Access results
    >>> amplitude = result['amplitude']  # Shape: (n_freqs, n_timepoints)
    >>> phase = result['phase']
    >>> power = result['power']
"""

import math
import numpy as np
from typing import Union, Tuple, Dict, List, Optional
from joblib import Parallel, delayed
import warnings


def compute_wavelet_transform(
    signal: np.ndarray,
    sampling_rate: float,
    frequencies: np.ndarray,
    n_cycles: Union[float, List[float]] = 7,
    output: str = 'amplitude',
    normalization: str = 'L1',
    standardize: bool = False,
    time_window: Optional[float] = None,
    edge_mode: str = 'same',
    num_cores: int = -1
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """
    Compute time-frequency decomposition using Morlet wavelets.
    
    This function applies wavelet transforms across multiple frequencies in parallel,
    providing flexible output options for different analysis needs.
    
    Parameters
    ----------
    signal : np.ndarray
        Input signal (typically LFP or EEG data). Shape: (n_timepoints,)
    sampling_rate : float
        Sampling rate of the signal in Hz
    frequencies : np.ndarray
        Array of frequencies to analyze in Hz. Shape: (n_freqs,)
    n_cycles : float or list of [min, max], default=7
        Number of cycles for the wavelet. Controls time-frequency tradeoff:
        - Low values (3-5): Better temporal resolution, worse frequency resolution
        - High values (7-14): Better frequency resolution, worse temporal resolution
        If list [min, max], cycles linearly interpolate from min to max across frequencies
    output : str, default='amplitude'
        Type of output to return:
        - 'complex': Complex wavelet coefficients
        - 'amplitude': Absolute value of coefficients
        - 'power': Squared amplitude (power)
        - 'phase': Phase angle in radians
        - 'all': Dictionary with all of the above plus standardized power
    normalization : str, default='L1'
        Wavelet normalization method:
        - 'L1': Sum of absolute values = 1 (preserves amplitude)
        - 'L2': Energy (sum of squares) = 1 (preserves power)
        - None: No normalization
    standardize : bool, default=False
        If True and output includes power, standardize by variance (z-score power)
    time_window : float, optional
        Explicit time window duration in seconds for wavelet.
        If None, automatically set to n_cycles/frequency
    edge_mode : str, default='same'
        How to handle edge effects:
        - 'same': Keep same length as input (edge artifacts present)
        - 'valid': Only keep valid convolution region (shorter output)
        - 'reflect': Reflect signal at edges before convolution
    num_cores : int, default=-1
        Number of CPU cores for parallel processing. -1 uses all available cores
    
    Returns
    -------
    result : np.ndarray or dict
        If output != 'all': np.ndarray of shape (n_freqs, n_timepoints)
        If output == 'all': Dictionary with keys:
            - 'complex': Complex coefficients
            - 'amplitude': Amplitude envelope
            - 'power': Power (amplitude squared)
            - 'phase': Phase in radians [-π, π]
            - 'z_power': Standardized power (if standardize=True)
            - 'frequencies': Frequency vector
            - 'n_cycles': Number of cycles used per frequency
    
    Notes
    -----
    Wavelet Design Considerations:
    - 3-cycle wavelets: Better for transient events, precise timing
    - 7-cycle wavelets: Better for sustained oscillations, precise frequency
    - For n_cycles > 7, verify wavelets taper to zero at edges
    
    Edge Effects:
    - Convolution introduces artifacts at signal boundaries
    - Consider trimming ~n_cycles/frequency seconds from each edge
    - Or use edge_mode='reflect' to minimize artifacts
    
    Memory Usage:
    - Output size: n_freqs × n_timepoints × dtype_size
    - For large signals, consider processing in chunks
    
    Examples
    --------
    >>> # Basic usage - get amplitude
    >>> amplitude = compute_wavelet_transform(signal, 1000, np.linspace(4, 100, 50))
    
    >>> # Adaptive cycles with full output
    >>> result = compute_wavelet_transform(
    ...     signal, 1000, np.linspace(4, 100, 50),
    ...     n_cycles=[3, 10], output='all'
    ... )
    >>> phase = result['phase']
    >>> power = result['power']
    
    >>> # High temporal resolution for event detection
    >>> amplitude = compute_wavelet_transform(
    ...     signal, 1000, np.linspace(4, 100, 50),
    ...     n_cycles=3, edge_mode='reflect'
    ... )
    """
    # Validate inputs
    signal = np.asarray(signal)
    if signal.ndim != 1:
        raise ValueError("Signal must be 1-dimensional")
    
    frequencies = np.asarray(frequencies)
    if frequencies.ndim != 1:
        raise ValueError("Frequencies must be 1-dimensional")
    
    if not isinstance(sampling_rate, (int, float)) or sampling_rate <= 0:
        raise ValueError("Sampling rate must be a positive number")
    
    valid_outputs = ['complex', 'amplitude', 'power', 'phase', 'all']
    if output not in valid_outputs:
        raise ValueError(f"output must be one of {valid_outputs}")
    
    valid_norms = ['L1', 'L2', None, 'none']
    if normalization not in valid_norms:
        raise ValueError(f"normalization must be one of {valid_norms}")
    if normalization == 'none':
        normalization = None
        
    # Parse n_cycles
    if isinstance(n_cycles, (int, float)):
        n_cycles_vec = np.full(len(frequencies), n_cycles)

    elif isinstance(n_cycles, (list, tuple, np.ndarray)):

        n_cycles = np.array(n_cycles)

        if n_cycles.ndim == 1 and len(n_cycles) == len(frequencies):
            # FULL CONTROL: user provides per-frequency cycles
            n_cycles_vec = n_cycles

        elif len(n_cycles) == 1:
            n_cycles_vec = np.full(len(frequencies), n_cycles[0])

        elif len(n_cycles) == 2:
            n_cycles_vec = np.linspace(n_cycles[0], n_cycles[1], len(frequencies))

        else:
            raise ValueError(
                "n_cycles must be:\n"
                "- scalar\n"
                "- [min, max]\n"
                "- or same length as frequencies"
            )

    else:
        raise ValueError("n_cycles must be a number, list, or array")

        
    
    # Handle edge effects
    if edge_mode == 'reflect':
        # Reflect signal at edges
        pad_samples = int(np.max(n_cycles_vec / frequencies.min()) * sampling_rate)
        signal = np.pad(signal, pad_samples, mode='reflect')
        trim_edges = True
    else:
        trim_edges = False
    
    # Compute wavelet transform in parallel
    complex_results = Parallel(n_jobs=num_cores)(
        delayed(_wavelet_transform_single)(
            signal, freq, sampling_rate, n_cyc, time_window, 
            normalization, edge_mode, standardize
        )
        for freq, n_cyc in zip(frequencies, n_cycles_vec)
    )
    
    complex_results = np.array(complex_results)
    
    # Trim reflected edges if necessary
    if trim_edges:
        complex_results = complex_results[:, pad_samples:-pad_samples]
    
    # Compute requested outputs
    if output == 'complex':
        return complex_results
    
    amplitude = np.abs(complex_results)
    
    if output == 'amplitude':
        return amplitude
    
    power = amplitude ** 2
    
    if output == 'power':
        return power
    
    if output == 'phase':
        return np.angle(complex_results)
    
    if output == 'all':
        result_dict = {
            'complex': complex_results,
            'amplitude': amplitude,
            'power': power,
            'phase': np.angle(complex_results),
            'frequencies': frequencies,
            'n_cycles': n_cycles_vec
        }
        
        # Add standardized power if requested
        if standardize:
            z_power = np.zeros_like(power)
            for i in range(len(frequencies)):
                std = np.nanstd(complex_results[i])
                if std > 0:
                    z_power[i] = power[i] / (std ** 2)
                else:
                    z_power[i] = np.nan
            result_dict['z_power'] = z_power
        
        return result_dict


import numpy as np

def make_theta_matched_cycles(
    frequencies,
    theta_freq,
    min_cycles=3,
    max_cycles=10,
    scaling='linear'  # or 'sqrt'
):
    """
    Generate n_cycles vector such that wavelet bandwidth matches theta modulation.

    Parameters
    ----------
    frequencies : array
        Frequencies used in wavelet transform
    theta_freq : float
        Center theta frequency (e.g., 6–10 Hz)
    min_cycles : float
        Lower bound for stability
    max_cycles : float
        Upper bound to avoid over-narrow filters
    scaling : str
        'linear'  -> n_cycles = f / theta
        'sqrt'    -> softer scaling (more stable)

    Returns
    -------
    n_cycles_vec : array
    """

    frequencies = np.asarray(frequencies)

    if scaling == 'linear':
        n_cycles_vec = frequencies / theta_freq

    elif scaling == 'sqrt':
        n_cycles_vec = np.sqrt(frequencies / theta_freq) * theta_freq

    else:
        raise ValueError("scaling must be 'linear' or 'sqrt'")

    # clip to reasonable range
    n_cycles_vec = np.clip(n_cycles_vec, min_cycles, max_cycles)

    return n_cycles_vec

def _wavelet_transform_single(
    signal: np.ndarray,
    frequency: float,
    sampling_rate: float,
    n_cycles: float,
    time_window: Optional[float],
    normalization: Optional[str],
    edge_mode: str,
    standardize: bool
) -> np.ndarray:
    """
    Apply single-frequency wavelet transform (internal function).
    
    This is called by the parallel processor for each frequency.
    """
    # Generate wavelet
    wavelet, _ = create_morlet_wavelet(
        frequency, n_cycles, sampling_rate, 
        time_window, normalization
    )
    
    # Convolve
    if edge_mode == 'valid':
        convolution_mode = 'valid'
    else:
        convolution_mode = 'same'
    
    complex_signal = np.convolve(signal, wavelet, mode=convolution_mode)
    
    return complex_signal


def create_morlet_wavelet(
    frequency: float,
    n_cycles: float = 7,
    sampling_rate: float = 1000,
    time_window: Optional[float] = None,
    normalization: Optional[str] = 'L1'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a complex Morlet wavelet kernel.
    
    The Morlet wavelet is a complex sinusoidal carrier modulated by a Gaussian envelope.
    It provides optimal time-frequency localization for oscillatory signals.
    
    Parameters
    ----------
    frequency : float
        Center frequency of the wavelet in Hz
    n_cycles : float, default=7
        Number of wavelet cycles. Controls time-frequency resolution tradeoff:
        - More cycles: Better frequency resolution, worse time resolution
        - Fewer cycles: Better time resolution, worse frequency resolution
    sampling_rate : float, default=1000
        Sampling rate in Hz
    time_window : float, optional
        Total duration of wavelet in seconds. If None, set to n_cycles/frequency
        (recommended to ensure proper wavelet decay)
    normalization : str or None, default='L1'
        Normalization method:
        - 'L1': Sum of absolute values = 1 (amplitude preserving)
        - 'L2': Sum of squared values = 1 (energy preserving)
        - None: No normalization (matches Gaussian amplitude factor)
    
    Returns
    -------
    wavelet : np.ndarray (complex)
        Complex Morlet wavelet kernel
    time_vector : np.ndarray
        Time vector in samples, centered at 0
    
    Notes
    -----
    The Morlet wavelet is defined as:
        ψ(t) = A * exp(-t²/(2σ²)) * exp(2πift)
    
    where:
        - σ = n_cycles / (2πf) is the Gaussian standard deviation
        - f is the center frequency
        - A is a normalization factor
    
    Examples
    --------
    >>> wavelet, time_vec = create_morlet_wavelet(10, n_cycles=5, sampling_rate=1000)
    >>> # wavelet is complex, time_vec in samples
    
    >>> # Visualize
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(time_vec, np.real(wavelet), label='Real')
    >>> plt.plot(time_vec, np.imag(wavelet), label='Imaginary')
    >>> plt.legend()
    """
    # Calculate Gaussian standard deviation
    sigma_time = n_cycles / (2 * np.pi * frequency)  # in seconds
    
    # Determine time window
    if time_window is None:
        time_window = n_cycles / frequency  # Default: one cycle per frequency
    
    # Create time vector centered at 0
    half_n_samples = int(np.round(time_window * sampling_rate / 2))
    sample_indices = np.arange(-half_n_samples, half_n_samples + 1)
    
    # Convert to time
    time_vec = sample_indices / sampling_rate
    
    # Gaussian envelope
    sigma_samples = sigma_time * sampling_rate
    dt = 1 / sampling_rate
    
    # Normalization factor for Gaussian
    A = dt / (np.sqrt(2 * np.pi) * sigma_samples)
    gaussian_envelope = A * np.exp(-(sample_indices ** 2) / (2 * sigma_samples ** 2))
    
    # Complex sinusoidal carrier
    carrier = np.exp(2j * np.pi * frequency * time_vec)
    
    # Morlet wavelet
    wavelet = gaussian_envelope * carrier
    
    # Apply normalization
    if normalization == 'L1':
        wavelet /= np.sum(np.abs(wavelet))
    elif normalization == 'L2':
        wavelet /= np.sqrt(np.sum(np.abs(wavelet) ** 2))
    
    return wavelet, sample_indices


def instantaneous_frequency(
    phase: np.ndarray,
    sampling_rate: float,
    unwrap: bool = True
) -> np.ndarray:
    """
    Compute instantaneous frequency from phase.
    
    Parameters
    ----------
    phase : np.ndarray
        Phase values in radians, shape (n_freqs, n_timepoints) or (n_timepoints,)
    sampling_rate : float
        Sampling rate in Hz
    unwrap : bool, default=True
        Whether to unwrap phase before computing derivative
    
    Returns
    -------
    inst_freq : np.ndarray
        Instantaneous frequency in Hz
    
    Notes
    -----
    Instantaneous frequency is the time derivative of phase:
        f(t) = (1/2π) * dφ/dt
    """
    if unwrap:
        phase = np.unwrap(phase, axis=-1)
    
    # Compute derivative (in radians/sample)
    phase_diff = np.diff(phase, axis=-1)
    
    # Convert to Hz
    inst_freq = phase_diff * sampling_rate / (2 * np.pi)
    
    # Pad to maintain original length
    if phase.ndim == 1:
        inst_freq = np.concatenate([[inst_freq[0]], inst_freq])
    else:
        inst_freq = np.concatenate([inst_freq[:, [0]], inst_freq], axis=1)
    
    return inst_freq


def phase_locking_value(
    phase1: np.ndarray,
    phase2: np.ndarray
) -> float:
    """
    Compute phase-locking value (PLV) between two signals.
    
    PLV measures phase synchronization between signals, ranging from 0 (no sync)
    to 1 (perfect synchronization).
    
    Parameters
    ----------
    phase1, phase2 : np.ndarray
        Phase time series in radians, shape (n_timepoints,)
    
    Returns
    -------
    plv : float
        Phase-locking value [0, 1]
    
    Examples
    --------
    >>> phase1 = result1['phase'][freq_idx, :]
    >>> phase2 = result2['phase'][freq_idx, :]
    >>> plv = phase_locking_value(phase1, phase2)
    """
    phase_diff = phase1 - phase2
    plv = np.abs(np.mean(np.exp(1j * phase_diff)))
    return plv


def get_edge_artifact_mask(
    n_timepoints: int,
    frequencies: np.ndarray,
    n_cycles: Union[float, np.ndarray],
    sampling_rate: float,
    threshold_cycles: float = 2
) -> np.ndarray:
    """
    Create a boolean mask indicating edge artifact regions.
    
    Edge artifacts occur where the wavelet extends beyond the signal boundaries.
    This function identifies these regions for masking or removal.
    
    Parameters
    ----------
    n_timepoints : int
        Length of the signal
    frequencies : np.ndarray
        Frequency vector used in analysis
    n_cycles : float or np.ndarray
        Number of cycles per frequency
    sampling_rate : float
        Sampling rate in Hz
    threshold_cycles : float, default=2
        Number of cycles to mark as artifact at each edge
    
    Returns
    -------
    mask : np.ndarray
        Boolean mask of shape (n_freqs, n_timepoints)
        True = reliable data, False = edge artifact
    
    Examples
    --------
    >>> mask = get_edge_artifact_mask(1000, freqs, n_cycles, 1000)
    >>> clean_amplitude = amplitude * mask
    """
    if isinstance(n_cycles, (int, float)):
        n_cycles = np.full(len(frequencies), n_cycles)
    
    mask = np.ones((len(frequencies), n_timepoints), dtype=bool)
    
    for i, (freq, cyc) in enumerate(zip(frequencies, n_cycles)):
        # Duration of artifact region
        artifact_duration = threshold_cycles / freq  # seconds
        artifact_samples = int(artifact_duration * sampling_rate)
        
        # Mark edges as artifacts
        mask[i, :artifact_samples] = False
        mask[i, -artifact_samples:] = False
    
    return mask


def plot_wavelet_family(
    frequencies: np.ndarray,
    n_cycles: Union[float, List[float]] = 7,
    sampling_rate: float = 1000,
    normalization: str = 'L1'
):
    """
    Visualize a family of wavelets across frequencies.
    
    Parameters
    ----------
    frequencies : np.ndarray
        Frequencies to visualize
    n_cycles : float or list
        Number of cycles per frequency
    sampling_rate : float
        Sampling rate in Hz
    normalization : str
        Normalization method
    
    Examples
    --------
    >>> plot_wavelet_family(np.array([5, 10, 20, 40]), n_cycles=[3, 10])
    """
    import matplotlib.pyplot as plt
    
    # Parse n_cycles
    if isinstance(n_cycles, (int, float)):
        n_cycles_vec = np.full(len(frequencies), n_cycles)
    elif len(n_cycles) == 2:
        n_cycles_vec = np.linspace(n_cycles[0], n_cycles[1], len(frequencies))
    else:
        n_cycles_vec = np.array(n_cycles)
    
    fig, axes = plt.subplots(len(frequencies), 2, figsize=(12, 2*len(frequencies)))
    if len(frequencies) == 1:
        axes = axes[np.newaxis, :]
    
    for i, (freq, n_cyc) in enumerate(zip(frequencies, n_cycles_vec)):
        wavelet, time_idx = create_morlet_wavelet(
            freq, n_cyc, sampling_rate, normalization=normalization
        )
        time_vec = time_idx / sampling_rate
        
        # Time domain
        axes[i, 0].plot(time_vec, np.real(wavelet), label='Real', alpha=0.7)
        axes[i, 0].plot(time_vec, np.imag(wavelet), label='Imag', alpha=0.7)
        axes[i, 0].plot(time_vec, np.abs(wavelet), 'k--', label='Envelope', linewidth=2)
        axes[i, 0].set_title(f'{freq:.1f} Hz, {n_cyc:.1f} cycles')
        axes[i, 0].set_xlabel('Time (s)')
        axes[i, 0].set_ylabel('Amplitude')
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)
        
        # Frequency domain
        fft_wavelet = np.fft.fft(wavelet)
        freqs_fft = np.fft.fftfreq(len(wavelet), 1/sampling_rate)
        pos_freqs = freqs_fft[:len(freqs_fft)//2]
        pos_fft = np.abs(fft_wavelet[:len(freqs_fft)//2])
        
        axes[i, 1].plot(pos_freqs, pos_fft)
        axes[i, 1].axvline(freq, color='r', linestyle='--', label=f'Center: {freq} Hz')
        axes[i, 1].set_xlabel('Frequency (Hz)')
        axes[i, 1].set_ylabel('Magnitude')
        axes[i, 1].set_xlim([0, min(100, sampling_rate/2)])
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# Backward compatibility aliases
def my_wavelet_parallel(*args, **kwargs):
    """Legacy function name. Use compute_wavelet_transform() instead."""
    warnings.warn(
        "my_wavelet_parallel is deprecated. Use compute_wavelet_transform() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    # Map old parameters to new ones
    if len(args) >= 3:
        signal, srate, freqvector = args[:3]
        kwargs.setdefault('sampling_rate', srate)
        kwargs.setdefault('frequencies', freqvector)
        return compute_wavelet_transform(signal, **kwargs)
    return compute_wavelet_transform(*args, **kwargs)


if __name__ == "__main__":
    # Quick test/demo
    print("Wavelet Transform Module")
    print("=" * 50)
    
    # Generate test signal
    srate = 1000
    duration = 2
    t = np.linspace(0, duration, int(srate * duration))
    
    # Signal with 10 Hz and 30 Hz components
    signal = (
        np.sin(2 * np.pi * 10 * t) +
        0.5 * np.sin(2 * np.pi * 30 * t) +
        0.1 * np.random.randn(len(t))
    )
    
    # Compute wavelet transform
    freqs = np.linspace(5, 50, 30)
    result = compute_wavelet_transform(
        signal, srate, freqs,
        n_cycles=[3, 10],
        output='all'
    )
    
    print(f"✓ Computed wavelet transform")
    print(f"  Frequencies: {len(freqs)}")
    print(f"  Signal length: {len(signal)}")
    print(f"  Output shape: {result['amplitude'].shape}")
    print(f"  Available outputs: {list(result.keys())}")
    print("\nModule ready for use!")
