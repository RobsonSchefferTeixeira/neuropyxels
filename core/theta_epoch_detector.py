"""
theta_epoch_detector.py

Theta epoch detection: find epochs where theta power (4-12 Hz) is higher
than delta power (1-4 Hz) using a sliding window approach.

The detection works as follows:
1. Bandpass filter the signal for theta (4-12 Hz) and delta (1-4 Hz) bands
2. Compute instantaneous power for each band (using Hilbert transform)
3. For each time point, compute the theta/delta power ratio
4. Find epochs where the ratio exceeds a threshold
5. Optionally merge nearby epochs and filter by minimum duration

No GUI dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert


@dataclass
class ThetaEpochParams:
    """Parameters for theta epoch detection."""
    theta_low: float = 4.0       # Theta band low (Hz)
    theta_high: float = 12.0     # Theta band high (Hz)
    delta_low: float = 1.0       # Delta band low (Hz)
    delta_high: float = 4.0      # Delta band high (Hz)
    ratio_threshold: float = 1.5  # Theta/delta power ratio threshold
    min_duration_ms: float = 500.0  # Minimum epoch duration (ms)
    merge_gap_ms: float = 500.0     # Merge epochs separated by this gap (ms)
    power_window_ms: float = 1000.0 # Window for averaging instantaneous band power
    power_overlap: float = 0.50     # Fractional overlap between power windows
    filter_order: int = 4
    use_csd: bool = False


@dataclass
class ThetaEpoch:
    """One detected theta epoch."""
    channel: int
    start_sample: int
    end_sample: int
    peak_sample: int
    mean_theta_power: float
    mean_delta_power: float
    mean_ratio: float
    peak_ratio: float
    duration_ms: float
    manual: bool = False


class ThetaEpochDetector:
    """Stateless theta epoch detection logic."""
    
    def apply_bandpass(self, signal: np.ndarray, sample_rate: float,
                       low_freq: float, high_freq: float,
                       order: int = 4) -> np.ndarray:
        """Apply bandpass filter to signal."""
        nyq = 0.5 * sample_rate
        if high_freq >= nyq:
            high_freq = nyq - 1.0
        sos = butter(order, [low_freq / nyq, high_freq / nyq],
                     btype="band", output="sos")
        return sosfiltfilt(sos, signal)
    
    def compute_band_power(self, signal: np.ndarray, sample_rate: float,
                           low_freq: float, high_freq: float,
                           order: int = 4) -> np.ndarray:
        """Compute instantaneous power for a frequency band."""
        filtered = self.apply_bandpass(signal, sample_rate, low_freq, high_freq, order)
        analytic = hilbert(filtered)
        return np.abs(analytic) ** 2
    
    def _window_average(self, values: np.ndarray, sample_rate: float,
                        window_ms: float, overlap: float) -> np.ndarray:
        """Estimate windowed mean power at every sample.

        Band power is first obtained with the Hilbert transform. This method
        then averages it in overlapping windows and linearly interpolates the
        window-center estimates back onto the original sample grid.
        """
        values = np.asarray(values, dtype=np.float64)
        n = len(values)
        if n == 0:
            return values.copy()

        window_samples = max(1, int(round(window_ms / 1000.0 * sample_rate)))
        window_samples = min(window_samples, n)
        overlap = float(np.clip(overlap, 0.0, 0.95))
        step = max(1, int(round(window_samples * (1.0 - overlap))))

        starts = np.arange(0, max(1, n - window_samples + 1), step, dtype=int)
        if starts[-1] != n - window_samples:
            starts = np.append(starts, n - window_samples)

        csum = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
        means = (csum[starts + window_samples] - csum[starts]) / window_samples
        centers = starts + (window_samples - 1) / 2.0

        if len(means) == 1:
            return np.full(n, means[0], dtype=np.float64)

        return np.interp(
            np.arange(n, dtype=np.float64),
            centers,
            means,
            left=means[0],
            right=means[-1],
        )

    def detect(self, signal: np.ndarray, sample_rate: float,
               params: ThetaEpochParams, channel: int = 0) -> list[ThetaEpoch]:
        """
        Detect theta epochs in a single channel signal.
        
        Parameters
        ----------
        signal : 1D array
            Raw or preprocessed signal for one channel
        sample_rate : float
            Sampling rate in Hz
        params : ThetaEpochParams
            Detection parameters
        channel : int
            Channel identifier for output
        
        Returns
        -------
        list[ThetaEpoch]
            Detected theta epochs, sorted by start_sample
        """
        signal = np.asarray(signal, dtype=np.float64)
        
        # Compute instantaneous theta/delta power, then average it in the
        # user-selected overlapping windows. The default is 1 s / 50%, a
        # conventional compromise for low-frequency band-power estimates.
        theta_power = self.compute_band_power(
            signal, sample_rate, params.theta_low, params.theta_high, params.filter_order
        )
        delta_power = self.compute_band_power(
            signal, sample_rate, params.delta_low, params.delta_high, params.filter_order
        )

        theta_power = self._window_average(
            theta_power, sample_rate, params.power_window_ms, params.power_overlap
        )
        delta_power = self._window_average(
            delta_power, sample_rate, params.power_window_ms, params.power_overlap
        )
        
        # Compute ratio (avoid division by zero)
        delta_power_safe = np.maximum(delta_power, 1e-10)
        ratio = theta_power / delta_power_safe
        
        # Find epochs where ratio exceeds threshold
        above_threshold = ratio > params.ratio_threshold
        
        if not np.any(above_threshold):
            return []
        
        # Find contiguous regions
        above_idx = np.where(above_threshold)[0]
        gaps = np.where(np.diff(above_idx) > 1)[0]
        run_starts = np.concatenate(([0], gaps + 1))
        run_ends = np.concatenate((gaps, [len(above_idx) - 1]))
        
        epochs = []
        for rs, re in zip(run_starts, run_ends):
            start_idx = above_idx[rs]
            end_idx = above_idx[re]
            
            # Check minimum duration
            duration_samples = end_idx - start_idx + 1
            duration_ms = duration_samples / sample_rate * 1000
            if duration_ms < params.min_duration_ms:
                continue
            
            # Find peak within epoch
            epoch_ratio = ratio[start_idx:end_idx + 1]
            peak_sample = start_idx + int(np.argmax(epoch_ratio))
            
            epochs.append(ThetaEpoch(
                channel=channel,
                start_sample=start_idx,
                end_sample=end_idx,
                peak_sample=peak_sample,
                mean_theta_power=float(np.mean(theta_power[start_idx:end_idx + 1])),
                mean_delta_power=float(np.mean(delta_power[start_idx:end_idx + 1])),
                mean_ratio=float(np.mean(epoch_ratio)),
                peak_ratio=float(np.max(epoch_ratio)),
                duration_ms=duration_ms,
                manual=False,
            ))
        
        # Merge nearby epochs if gap is small
        if params.merge_gap_ms > 0:
            epochs = self._merge_epochs(epochs, sample_rate, params.merge_gap_ms)
        
        # Sort by start sample
        epochs.sort(key=lambda e: e.start_sample)
        
        return epochs
    
    def _merge_epochs(self, epochs: list[ThetaEpoch], sample_rate: float,
                      merge_gap_ms: float) -> list[ThetaEpoch]:
        """Merge epochs that are separated by less than merge_gap_ms."""
        if len(epochs) <= 1:
            return epochs
        
        gap_samples = int(merge_gap_ms / 1000 * sample_rate)
        
        merged = [epochs[0]]
        for epoch in epochs[1:]:
            prev = merged[-1]
            gap = epoch.start_sample - prev.end_sample
            
            if gap <= gap_samples:
                # Merge epochs
                merged[-1] = ThetaEpoch(
                    channel=prev.channel,
                    start_sample=prev.start_sample,
                    end_sample=epoch.end_sample,
                    peak_sample=prev.peak_sample if prev.peak_ratio >= epoch.peak_ratio else epoch.peak_sample,
                    mean_theta_power=(prev.mean_theta_power + epoch.mean_theta_power) / 2,
                    mean_delta_power=(prev.mean_delta_power + epoch.mean_delta_power) / 2,
                    mean_ratio=(prev.mean_ratio + epoch.mean_ratio) / 2,
                    peak_ratio=max(prev.peak_ratio, epoch.peak_ratio),
                    duration_ms=(epoch.end_sample - prev.start_sample) / sample_rate * 1000,
                    manual=False,
                )
            else:
                merged.append(epoch)
        
        return merged