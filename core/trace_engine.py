"""
trace_engine.py

Data loading, filtering, and downsampling logic for raw Neuropixels traces,
extracted from neuropixels_viewer.py's NeuropixelsViewer class.

This is a deliberate 1:1 behavioral port of that class's data-handling
methods (load_data_file, get_data_segment, apply_filter,
get_filtered_segment, load_spikes, get_spikes_in_window) with all
tkinter/matplotlib GUI code removed. The filtering approach here is NOT
the same as core/filters.py's bandpass_filter -- this uses a causal
sosfilt with manual transient blanking (not zero-phase sosfiltfilt), to
match exactly what the existing viewer has been producing. If you want
zero-phase filtering here instead, that's a real behavior change worth
deciding on deliberately, not something to silently switch.

No GUI dependency. Safe to import from either a Qt widget or a script.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from scipy.signal import butter, sosfilt, iirnotch, filtfilt


class TraceEngine:
    """
    Owns the raw data memmap, spike data, and filter/downsample settings
    for one continuous.dat recording. No plotting, no widgets.
    """

    def __init__(self, n_channels: int = 384, sample_rate: float = 30000, dtype=np.int16):
        self.n_channels = n_channels
        self.sr = sample_rate
        self.dtype = dtype

        self.data_file: Optional[Path] = None
        self.data: Optional[np.memmap] = None
        self.total_samples = 0
        self.total_duration = 0.0
        self.data_loaded = False

        # Channel selection
        self.current_channels: list[int] = list(range(min(8, n_channels)))
        
        # Channel depths (y coordinates) - used for sorting
        self.channel_depths: dict[int, float] = {}

        # Filter parameters -- defaults match neuropixels_viewer.py
        self.filter_low = 300
        self.filter_high = 6000
        self.filter_order = 4
        self.notch_freq = 50
        self.notch_enabled = False
        self.filter_enabled = True

        # Downsampling
        self.downsample_factor = 1
        self.max_display_samples = 50000
        self.auto_downsample = True
        self.current_downsample = 1

        # Spikes
        self.spike_times: Optional[np.ndarray] = None
        self.spike_clusters: Optional[np.ndarray] = None
        self.cluster_ids: Optional[np.ndarray] = None
        self.loaded_spikes = False
        self.spikes_dir: Optional[Path] = None

        # Timestamp attributes
        self.timestamps: Optional[np.ndarray] = None
        self.timestamps_loaded = False
        self.timestamp_start = 0.0
        self.timestamp_end = 0.0
        self.timestamp_duration = 0.0


    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_timestamps(self, timestamps_path: Path) -> bool:
        """Load a timestamps.npy file."""
        timestamps_path = Path(timestamps_path)
        if not timestamps_path.exists():
            return False
        
        try:
            self.timestamps = np.load(timestamps_path).flatten()
            
            # Validate that timestamps match data length
            if len(self.timestamps) != self.total_samples:
                print(f"Warning: Timestamps length ({len(self.timestamps)}) "
                    f"doesn't match data length ({self.total_samples}). "
                    f"Using sampling rate for time calculation.")
                self.timestamps = None
                self.timestamps_loaded = False
                return False
            
            self.timestamps_loaded = True
            
            # Store the first timestamp as the start time reference
            self.timestamp_start = float(self.timestamps[0])
            self.timestamp_end = float(self.timestamps[-1])
            self.timestamp_duration = self.timestamp_end - self.timestamp_start
            
            return True
        except Exception as e:
            print(f"Error loading timestamps: {e}")
            return False
            
    def load_data_file(self, file_path) -> None:
        """Memory-map a continuous.dat file."""
        self.data_file = Path(file_path)
        if not self.data_file.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_file}")

        file_size = self.data_file.stat().st_size
        bytes_per_sample = np.dtype(self.dtype).itemsize
        self.total_samples = file_size // (self.n_channels * bytes_per_sample)
        if self.total_samples == 0:
            raise ValueError(
                f"File {self.data_file} is too small for {self.n_channels} "
                f"channels at {bytes_per_sample} bytes/sample."
            )
        self.total_duration = self.total_samples / self.sr

        self.data = np.memmap(
            self.data_file, dtype=self.dtype, mode="r",
            shape=(self.total_samples, self.n_channels),
        )
        self.data_loaded = True

    def load_spikes(self, spikes_dir) -> bool:
        """Load Kilosort spike_times.npy / spike_clusters.npy if present."""
        self.spikes_dir = Path(spikes_dir)

        spike_times_file = self.spikes_dir / "spike_times.npy"
        if not spike_times_file.exists():
            return False
        self.spike_times = np.load(spike_times_file).flatten()

        spike_clusters_file = self.spikes_dir / "spike_clusters.npy"
        if spike_clusters_file.exists():
            self.spike_clusters = np.load(spike_clusters_file).flatten()
            self.cluster_ids = np.unique(self.spike_clusters)

        self.loaded_spikes = True
        return True

    # ------------------------------------------------------------------
    # Channel selection and depths
    # ------------------------------------------------------------------

    def set_channels(self, channels: list[int]):
        """Set which channels to display."""
        self.current_channels = list(channels)

    def set_channel_depths(self, depths: dict[int, float]):
        """Set the depth (Y coordinate) for each channel."""
        self.channel_depths = depths

    def get_channel_depths(self) -> dict[int, float]:
        """Get the depth map for channels."""
        return self.channel_depths

    def get_sorted_channels_by_depth(self) -> list[int]:
        """Get channels sorted by depth (ascending)."""
        if self.channel_depths:
            return sorted(
                self.current_channels,
                key=lambda ch: self.channel_depths.get(ch, 0.0)
            )
        return list(self.current_channels)

    # ------------------------------------------------------------------
    # Raw segment access
    # ------------------------------------------------------------------

    def get_data_segment(self, start_time: float, duration: float,
                        channels: Optional[list[int]] = None):
        """Returns (segment, start_idx, end_idx) with segment shaped
        (n_channels, n_samples), or (None, None, None) if no data loaded."""
        if not self.data_loaded:
            return None, None, None
        
        if self.timestamps_loaded and self.timestamps is not None:
            end_time = start_time + duration
            start_idx = np.searchsorted(self.timestamps, start_time, side='left')
            end_idx = np.searchsorted(self.timestamps, end_time, side='right')
        else:
            start_idx = max(0, int(start_time * self.sr))
            end_idx = min(self.total_samples, int((start_time + duration) * self.sr))
        
        if channels is not None:
            segment = self.data[start_idx:end_idx, channels].T
        else:
            segment = self.data[start_idx:end_idx, :].T
        return segment, start_idx, end_idx

    def get_timestamps_in_range(self, start_time: float, end_time: float) -> np.ndarray:
        """Get the timestamps within a time range."""
        if self.timestamps_loaded and self.timestamps is not None:
            start_idx = np.searchsorted(self.timestamps, start_time, side='left')
            end_idx = np.searchsorted(self.timestamps, end_time, side='right')
            return self.timestamps[start_idx:end_idx]
        else:
            # Generate from sampling rate
            start_idx = int(start_time * self.sr)
            end_idx = int(end_time * self.sr)
            start_idx = max(0, start_idx)
            end_idx = min(self.total_samples, end_idx)
            return np.arange(start_idx, end_idx) / self.sr
            
    # ------------------------------------------------------------------
    # Filtering (causal sosfilt + manual transient blanking)
    # ------------------------------------------------------------------

    def apply_filter(self, data: np.ndarray) -> np.ndarray:
        if not self.filter_enabled or not self.data_loaded:
            return data

        nyq = 0.5 * self.sr

        order = min(3, self.filter_order) if self.filter_low < 10 else self.filter_order

        sos = butter(order, [self.filter_low / nyq, self.filter_high / nyq],
                     btype="band", output="sos")
        filtered = sosfilt(sos, data, axis=1)

        transient_samples = int(0.1 * self.sr)  # 100ms
        if filtered.shape[1] > transient_samples:
            filtered[:, :transient_samples] = filtered[:, transient_samples:transient_samples + 1]

        if self.notch_enabled and self.notch_freq is not None:
            w0 = self.notch_freq / nyq
            if 0 < w0 < 1:
                b_notch, a_notch = iirnotch(w0, Q=30)
                filtered = filtfilt(b_notch, a_notch, filtered, axis=1)

        return filtered

    def get_filtered_segment(self, start_time: float, duration: float,
                              channels: Optional[list[int]] = None):
        """Returns (times, filtered_data) with filtered_data shaped
        (n_channels, n_samples_out), downsampled per current settings."""
        if not self.data_loaded:
            return None, None

        if channels is None:
            channels = self.current_channels

        raw_segment, _, _ = self.get_data_segment(start_time, duration, channels)
        if raw_segment is None:
            return None, None

        filtered_data = self.apply_filter(raw_segment)

        n_samples = filtered_data.shape[1]
        if self.auto_downsample and self.downsample_factor == 1 and n_samples > self.max_display_samples:
            downsample_factor = max(1, int(np.ceil(n_samples / self.max_display_samples)))
        else:
            downsample_factor = max(1, self.downsample_factor)

        if downsample_factor > 1:
            filtered_data = filtered_data[:, ::downsample_factor]
            self.current_downsample = downsample_factor
        else:
            self.current_downsample = 1

        n_samples_out = filtered_data.shape[1]
        times = np.linspace(start_time, start_time + duration, n_samples_out)

        return times, filtered_data

    # ------------------------------------------------------------------
    # Spikes
    # ------------------------------------------------------------------

    def get_spikes_in_window(self, start_time: float, duration: float) -> np.ndarray:
        if not self.loaded_spikes or not self.data_loaded:
            return np.array([])

        spike_times_sec = self.spike_times / self.sr
        mask = (spike_times_sec >= start_time) & (spike_times_sec <= start_time + duration)
        return spike_times_sec[mask]

    def get_time_window_sample_range(self, start_time: float,
                                      end_time: float) -> tuple[int, int]:
        """
        Resolve a [start_time, end_time) window (in the same time basis
        used everywhere else -- absolute timestamp seconds when
        timestamps are loaded, elapsed seconds since recording start
        otherwise) to a clipped [start_idx, end_idx) sample range.

        Pulled out of get_channel_data() so callers that need to know
        exactly which samples were returned (e.g. the trace view, to
        convert each plotted sample back to its true time via
        sample_to_time()) don't have to re-derive start_idx themselves
        with a second, potentially-drifting implementation of this same
        timestamps-vs-sample-rate branch.
        """
        if self.timestamps_loaded and self.timestamps is not None:
            start_idx = np.searchsorted(self.timestamps, start_time, side='left')
            end_idx = np.searchsorted(self.timestamps, end_time, side='right')
        else:
            start_idx = int(start_time * self.sr)
            end_idx = int(end_time * self.sr)

        start_idx = max(0, int(start_idx))
        end_idx = min(self.data.shape[0] if self.data is not None else 0, int(end_idx))
        return start_idx, end_idx

    def sample_to_time(self, sample: int) -> float:
        """Convert an absolute sample index to its time value, in the
        same basis as get_channel_data's start_time/end_time (absolute
        timestamp seconds if timestamps are loaded, else elapsed seconds
        since recording start)."""
        if self.timestamps_loaded and self.timestamps is not None:
            idx = max(0, min(int(sample), len(self.timestamps) - 1))
            return float(self.timestamps[idx])
        return sample / self.sr

    def get_channel_data(self, channel: int, start_time: float, 
                        end_time: float) -> np.ndarray:
        """
        Get data for a single channel between start_time and end_time.
        Uses timestamps if available, otherwise uses sampling rate.
        """
        if not self.data_loaded:
            raise ValueError("No data loaded")
        
        if channel < 0 or channel >= self.n_channels:
            raise ValueError(f"Channel {channel} out of range")

        start_idx, end_idx = self.get_time_window_sample_range(start_time, end_time)
        if start_idx >= end_idx:
            return np.array([])
        return self.data[start_idx:end_idx, channel]



    def timestamp_to_sample(self, timestamp: float) -> int:
        """Convert a timestamp value to a sample index."""
        if self.timestamps_loaded and self.timestamps is not None:
            return int(np.searchsorted(self.timestamps, timestamp, side='left'))
        else:
            return int(timestamp * self.sr)



    def get_time_range(self) -> tuple[float, float]:
        """Get the valid time range for the loaded data."""
        if self.timestamps_loaded and self.timestamps is not None:
            return (self.timestamp_start, self.timestamp_end)
        else:
            return (0.0, self.total_duration)