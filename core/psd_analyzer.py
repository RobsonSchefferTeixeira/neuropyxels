"""
psd_analyzer.py

Power spectral density for one channel of a Neuropixels recording, using
scipy.signal.welch / periodogram under the hood. Kept separate from the
GUI so it can be called from a script or from another analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.signal import welch, periodogram


@dataclass
class PsdParams:
    """All tunable parameters for one PSD run."""
    channel: int
    start_time: float = 0.0
    end_time: float = 10.0
    use_whole_recording: bool = False
    method: str = "welch"           # "welch" or "periodogram"
    window: str = "hann"
    nperseg: int = 4096
    noverlap: int = 2048
    nfft: int = 8192
    detrend: str = "constant"
    target_sr: Optional[float] = 1000.0  # None or <= 0 = no downsampling


class PsdAnalyzer:
    """Compute a PSD for one channel over a chosen time window."""

    def __init__(self, engine):
        """
        engine : TraceEngine
            The loaded recording. PsdAnalyzer only reads
            engine.data / engine.sr / engine.total_duration.
        """
        self.engine = engine

    def compute(self, params: PsdParams) -> dict:
        """
        Returns a dict with:
            frequencies : ndarray (n_freq,)
            psd         : ndarray (n_freq,)
            channel     : int
            start_time  : float
            end_time    : float
            sample_rate : float
            n_samples   : int
            params      : PsdParams
        """
        if not self.engine.data_loaded:
            raise ValueError("No data loaded.")

        channel = int(params.channel)
        if channel < 0 or channel >= self.engine.n_channels:
            raise ValueError(f"Channel {channel} out of range.")

        if params.use_whole_recording:
            start_time = 0.0
            end_time = float(self.engine.total_duration)
        else:
            start_time = float(params.start_time)
            end_time = float(params.end_time)

        if end_time <= start_time:
            raise ValueError("End time must be greater than start time.")

        raw = self.engine.get_channel_data(channel, start_time, end_time)
        raw = np.asarray(raw, dtype=np.float64)
        if raw.size < 2:
            raise ValueError("Not enough samples in the selected window.")

        finite = np.isfinite(raw)
        if not np.any(finite):
            raise ValueError("All samples in the selected window are invalid.")
        if not np.all(finite):
            fill = float(np.mean(raw[finite]))
            raw = np.where(finite, raw, fill)

        sr = float(self.engine.sr)

        # ---- Optional downsampling before the PSD ----
        if params.target_sr is not None and params.target_sr > 0 and sr > params.target_sr:
            from scipy.signal import decimate
            dec_factor = int(round(sr / params.target_sr))
            if dec_factor >= 2:
                try:
                    raw = decimate(raw, dec_factor, ftype="fir")
                    sr = sr / dec_factor
                except Exception:
                    # If decimation fails for any reason (segment too short,
                    # numerical issue), fall through with the original
                    # signal and rate rather than crashing the whole PSD.
                    pass

        # Guard the parameters against the actual segment length: scipy
        # raises if nperseg > len(x), and noverlap must be < nperseg.
        n_samples = raw.size
        nperseg = int(min(max(8, params.nperseg), n_samples))
        noverlap = int(min(max(0, params.noverlap), nperseg - 1))
        nfft = int(max(nperseg, params.nfft))

        if params.method == "welch":
            freqs, psd = welch(
                raw,
                fs=sr,
                window=params.window,
                nperseg=nperseg,
                noverlap=noverlap,
                nfft=nfft,
                detrend=params.detrend,
                scaling="density",
                average="mean",
            )
        elif params.method == "periodogram":
            freqs, psd = periodogram(
                raw,
                fs=sr,
                window=params.window,
                nfft=nfft,
                detrend=params.detrend,
                scaling="density",
            )
        else:
            raise ValueError(f"Unknown method: {params.method!r}")

        return {
            "frequencies": freqs,
            "psd": psd,
            "channel": channel,
            "start_time": start_time,
            "end_time": end_time,
            "sample_rate": float(sr),
            "n_samples": int(n_samples),
            "params": params,
        }