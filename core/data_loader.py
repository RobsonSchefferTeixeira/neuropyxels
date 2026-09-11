import os
import numpy as np
import scipy.io
import scipy.signal as sig
from scipy import signal
import matplotlib.pyplot as plt
import scipy.stats as stats
import time
import h5py
import xml.etree.ElementTree as ET
from scipy.io import loadmat
import glob as glob
import xml.etree.ElementTree as ET
from pathlib import Path
import warnings
from typing import Dict, List, Tuple, Optional, Union
import json

path = r'D:\_work_rscheffer\Animal_Data\RST'
scripts_path = path + '/scripts/'
figures_path = path + '/figures/'
results_path = path + '/results/'
data_path = path + '/data/'

bit_volts = 0.1949999928

import core.filters as filtering


class NeuropixelsDataLoader:
    """Class to handle loading and processing Neuropixels data based on parsed settings."""
    
    def __init__(self, base_path: str, folder: str, session: str, probe_info: Dict):
        """
        Initialize the data loader with parsed probe information.
        
        Args:
            folder: Animal folder name (e.g., 'ADM009')
            session: Session name (e.g., 'd20260111')
            probe_info: Dictionary from parse_neuropixels_settings
        """
        self.folder = folder
        self.session = session
        self.probe_info = probe_info
        # self.base_data_path = base_data_path
        
        # Base data path (adjust this based on your structure)
        # You might want to pass this as a parameter or define it globally
        self.base_data_path = Path(base_path) / 'data' / folder / f"{folder}_{session}" / f"{folder}_{session}_recording"
        
        # Find the recording structure
        self.recording_structure = self._find_recording_structure()
    
    def _find_recording_structure(self) -> Dict:
        """
        Find the actual recording structure in the file system.
        
        Returns:
            Dictionary with paths to the continuous data folder
        """
        if not self.base_data_path.exists():
            raise FileNotFoundError(f"Data path not found: {self.base_data_path}")
        
        # Look for Record Node folder
        record_node_dirs = [d for d in self.base_data_path.iterdir() 
                           if d.is_dir() and d.name.startswith('Record Node')]
        
        if not record_node_dirs:
            raise FileNotFoundError(f"No Record Node folders found in {self.base_data_path}")
        
        # Use the first Record Node folder (or you could match with nodeId if needed)
        record_node_dir = record_node_dirs[0]
        
        # Look for experiment folder
        experiment_dirs = [d for d in record_node_dir.iterdir() 
                          if d.is_dir() and d.name.startswith('experiment')]
        
        if not experiment_dirs:
            raise FileNotFoundError(f"No experiment folders found in {record_node_dir}")
        
        experiment_dir = experiment_dirs[0]
        
        # Look for recording folder
        recording_dirs = [d for d in experiment_dir.iterdir() 
                         if d.is_dir() and d.name.startswith('recording')]
        
        if not recording_dirs:
            raise FileNotFoundError(f"No recording folders found in {experiment_dir}")
        
        recording_dir = recording_dirs[0]
        
        # Construct the continuous data path
        continuous_path = recording_dir / 'continuous'
        events_path = recording_dir / 'events'


        
        return {
            'record_node': record_node_dir.name,
            'experiment': experiment_dir.name,
            'recording': recording_dir.name,
            'continuous_path': continuous_path,
            'events_path': events_path,

        }


    def load_events_ttls(self, probe_key: str) -> Path:

        if probe_key not in self.probe_info['probes']:
            raise ValueError(f"Probe {probe_key} not found. Available: {list(self.probe_info['probes'].keys())}")

        probe_data = self.probe_info['probes'][probe_key]
        folder_name = probe_data['folder_name']
        
        data_folder = self.recording_structure['events_path'] / folder_name
    
        states_file = data_folder /  'TTL' /'states.npy'
        timestamps_file = data_folder / 'TTL' / 'timestamps.npy'
        states = np.load(states_file)
        timestamps = np.load(timestamps_file)

        return states, timestamps


    def get_probe_data_folder(self, probe_key: str) -> Path:
        """
        Get the data folder path for a specific probe.
        
        Args:
            probe_key: Probe identifier (e.g., 'A_AP', 'B_AP', 'B_LFP')
            
        Returns:
            Path to the probe's data folder
        """
        if probe_key not in self.probe_info['probes']:
            raise ValueError(f"Probe {probe_key} not found. Available: {list(self.probe_info['probes'].keys())}")
        
        probe_data = self.probe_info['probes'][probe_key]
        folder_name = probe_data['folder_name']
        
        data_folder = self.recording_structure['continuous_path'] / folder_name
        
        if not data_folder.exists():
            raise FileNotFoundError(f"Data folder not found: {data_folder}")
        
        return data_folder
    
    def load_raw_data(self, probe_key: str, method: str = 'efficient_memory') -> Tuple[np.ndarray, np.ndarray]:
        """
        Load raw data from a probe.
        
        Args:
            probe_key: Probe identifier (e.g., 'A_AP', 'B_AP', 'B_LFP')
            method: Loading method ('efficient_memory' or 'efficient_speed')
            
        Returns:
            data: Raw data with shape (channels, samples)
            timestamps: Timestamps array
        """
        data_folder = self.get_probe_data_folder(probe_key)
        data_file = data_folder / 'continuous.dat'
        timestamps_file = data_folder / 'timestamps.npy'
        
        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")
        if not timestamps_file.exists():
            raise FileNotFoundError(f"Timestamps file not found: {timestamps_file}")
        
        probe_data = self.probe_info['probes'][probe_key]
        channel_count = probe_data['channel_count']
        data_format = np.int16
        
        # Load raw data
        if method == 'efficient_memory':
            # Memory-mapped file for large datasets
            file_size = np.memmap(data_file, dtype=data_format, mode='r').size
            n_samples = file_size // channel_count
            data = np.memmap(data_file, dtype=data_format, mode='r', shape=(n_samples, channel_count)).T
        elif method == 'efficient_speed':
            # Load all data at once
            data = np.fromfile(data_file, dtype=data_format)
            n_samples = data.size // channel_count
            data = data.reshape((n_samples, channel_count)).T
        else:
            raise ValueError(f"Unknown method '{method}'. Choose 'efficient_memory' or 'efficient_speed'.")
        
        # Load timestamps
        timestamps = np.load(timestamps_file)
        
        return data, timestamps

    def select_channels_by_coordinates(self, probe_key: str,
                                  x_range: Optional[Tuple[int, int]] = None,
                                  y_range: Optional[Tuple[int, int]] = None,
                                  x_list: Optional[List[int]] = None,
                                  y_list: Optional[List[int]] = None,
                                  channel_indices: Optional[List[int]] = None,
                                  x_step: Optional[int] = None,
                                  y_step: Optional[int] = None,
                                  reject_channels: Optional[List[int]] = None,
                                  reject_mode: str = 'ignore',
                                  return_coordinates: bool = False) -> Union[List[int], Tuple[List[int], Dict[str, List[float]]]]:
        """
        Select channels based on electrode coordinates.

        Args:
            probe_key: Probe identifier
            x_range: Tuple of (min_x, max_x) to select channels within x range
            y_range: Tuple of (min_y, max_y) to select channels within y range
            x_list: List of specific x coordinates to select
            y_list: List of specific y coordinates to select
            channel_indices: List of specific channel indices to select
            x_step: Step size for x coordinates (used with x_range)
            y_step: Step size for y coordinates (used with y_range)
            reject_channels: List of channel numbers to exclude.  If None or
                empty, no channels are rejected (default behaviour).
            reject_mode: How to handle rejected channels.
                'ignore'  – simply drop them from the selection (default).
                            The depth gap is preserved (useful when the CSD
                            NaN-safe kernel will handle it).
                'replace' – substitute each rejected channel with the nearest
                            valid channel on the same shank at the same depth
                            (same y).  If no same-depth candidate exists,
                            falls back to the nearest y on the same shank.
                            The replacement channel appears in the returned
                            list in place of the rejected one.
            return_coordinates: If True, returns (channel_indices, coordinates_dict)

        Returns:
            If return_coordinates=False: List of channel indices (0-based)
            If return_coordinates=True: Tuple of (channel_indices, {'x': [...], 'y': [...]})
        """
        probe_data = self.probe_info['probes'][probe_key]
        coords = probe_data.get('coordinates', {})
        
        if not coords.get('x') or not coords.get('y'):
            raise ValueError(f"No coordinate information available for probe {probe_key}")
        
        # Convert to numpy arrays
        x_arr = np.array(coords['x'])
        y_arr = np.array(coords['y'])
        ch_arr = np.array(coords.get('channels', list(range(len(x_arr)))))
        
        # Start with all channels
        mask = np.ones(len(ch_arr), dtype=bool)
        
        # Apply basic range filters
        if x_range is not None:
            mask &= (x_arr >= x_range[0]) & (x_arr <= x_range[1])
        
        if y_range is not None:
            mask &= (y_arr >= y_range[0]) & (y_arr <= y_range[1])
        
        if x_list is not None:
            mask &= np.isin(x_arr, x_list)
        
        if y_list is not None:
            mask &= np.isin(y_arr, y_list)
        
        if channel_indices is not None:
            mask &= np.isin(ch_arr, channel_indices)
        
        # Apply step selection if specified
        if x_step is not None or y_step is not None:
            # Get candidate channels after basic filtering
            candidate_indices = np.where(mask)[0]
            
            if len(candidate_indices) == 0:
                return [] if not return_coordinates else ([], {'x': [], 'y': []})
            
            x_candidates = x_arr[candidate_indices]
            y_candidates = y_arr[candidate_indices]
            ch_candidates = ch_arr[candidate_indices]
            
            # Get unique x positions in the candidates
            unique_x = np.unique(x_candidates)
            unique_y = np.unique(y_candidates)
            
            # Generate target positions
            if x_step is not None and x_range is not None:
                target_x_positions = np.arange(x_range[0], x_range[1] + x_step, x_step)
            elif x_step is not None:
                target_x_positions = np.arange(unique_x.min(), unique_x.max() + x_step, x_step)
            else:
                target_x_positions = None
            
            if y_step is not None and y_range is not None:
                target_y_positions = np.arange(y_range[0], y_range[1], y_step)
            elif y_step is not None:
                target_y_positions = np.arange(unique_y.min(), unique_y.max() + y_step, y_step)
            else:
                target_y_positions = None
            
            selected_indices_set = set()
            
            if target_x_positions is not None and target_y_positions is not None:
                # Both x and y stepping: create grid
                for target_x in target_x_positions:
                    for target_y in target_y_positions:
                        distances = np.sqrt((x_candidates - target_x)**2 + (y_candidates - target_y)**2)
                        nearest_idx = np.argmin(distances)
                        selected_indices_set.add(int(ch_candidates[nearest_idx]))
            
            elif target_x_positions is not None and target_y_positions is None:
                # Only x stepping: select for each x step, keep all y values within range
                for target_x in target_x_positions:
                    # Find electrodes with x closest to target_x
                    x_distances = np.abs(x_candidates - target_x)
                    min_x_dist = x_distances.min()
                    # Select all electrodes at that x distance (could be multiple if multiple electrodes share same x)
                    x_mask = x_distances == min_x_dist
                    for idx in np.where(x_mask)[0]:
                        selected_indices_set.add(int(ch_candidates[idx]))
            
            elif target_x_positions is None and target_y_positions is not None:
                # Only y stepping: select for each y step, keep all x values within range
                for target_y in target_y_positions:
                    # Find electrodes with y closest to target_y
                    y_distances = np.abs(y_candidates - target_y)
                    min_y_dist = y_distances.min()
                    # Select all electrodes at that y distance
                    y_mask = y_distances == min_y_dist
                    for idx in np.where(y_mask)[0]:
                        selected_indices_set.add(int(ch_candidates[idx]))
            
            # Create mask for selected channels
            mask = np.isin(ch_arr, list(selected_indices_set))
        
        # Get selected channel indices (before reject handling)
        selected_channels = ch_arr[mask]
        selected_indices = [int(ch) for ch in selected_channels]

        # ── Reject-channel handling ────────────────────────────────────────
        reject_set = set(reject_channels) if reject_channels else set()
        if reject_set:
            # Pull full probe coordinates for replacement look-ups
            probe_data_full = self.probe_info['probes'][probe_key]
            coords_full = probe_data_full.get('coordinates', {})
            all_x   = np.array(coords_full['x'])
            all_y   = np.array(coords_full['y'])
            all_ch  = np.array(coords_full.get('channels', list(range(len(all_x)))))
            shank_ids_full = np.array(
                probe_data_full.get('shanks', {}).get('shank_ids',
                                                       [0]*len(all_ch))
            )

            if reject_mode == 'ignore':
                # Simply drop the rejected channels
                selected_indices = [ch for ch in selected_indices
                                    if ch not in reject_set]

            elif reject_mode == 'replace':
                # Build a mapping: rejected ch → best replacement from the
                # full probe (same shank, same y preferred; nearest y fallback)
                patched = []
                already_used = set(selected_indices) - reject_set  # keep track to avoid duplicates
                for ch in selected_indices:
                    if ch not in reject_set:
                        patched.append(ch)
                        continue

                    # Find index of this channel in the full electrode list
                    ch_idx = np.where(all_ch == ch)[0]
                    if len(ch_idx) == 0:
                        # Channel not in coordinate list – just drop it
                        warnings.warn(
                            f"Rejected channel {ch} has no coordinate entry; "
                            "it will be dropped.")
                        continue
                    ch_idx = ch_idx[0]
                    ch_shank = shank_ids_full[ch_idx]
                    ch_y     = all_y[ch_idx]

                    # Candidate pool: same shank, not already selected/rejected
                    same_shank = (shank_ids_full == ch_shank)
                    not_rejected = ~np.isin(all_ch, list(reject_set))
                    not_used     = ~np.isin(all_ch, list(already_used))
                    pool_mask    = same_shank & not_rejected & not_used

                    if not pool_mask.any():
                        # Widen: allow already-selected (avoid total loss)
                        pool_mask = same_shank & not_rejected
                    if not pool_mask.any():
                        warnings.warn(
                            f"No replacement found for rejected channel {ch}; "
                            "it will be dropped.")
                        continue

                    pool_y    = all_y[pool_mask]
                    pool_ch   = all_ch[pool_mask]

                    # Prefer same depth, then nearest depth
                    same_depth = (pool_y == ch_y)
                    if same_depth.any():
                        # Among same-depth candidates pick the x-nearest
                        pool_x       = all_x[pool_mask]
                        ch_x         = all_x[ch_idx]
                        same_depth_x = pool_x[same_depth]
                        same_depth_c = pool_ch[same_depth]
                        best = same_depth_c[np.argmin(np.abs(same_depth_x - ch_x))]
                    else:
                        # Nearest depth on the same shank
                        best = pool_ch[np.argmin(np.abs(pool_y - ch_y))]

                    replacement = int(best)
                    already_used.add(replacement)
                    patched.append(replacement)
                    warnings.warn(
                        f"Channel {ch} rejected and replaced by channel "
                        f"{replacement} "
                        f"(y={float(all_y[all_ch==replacement][0]):.0f} µm, "
                        f"x={float(all_x[all_ch==replacement][0]):.0f} µm).")

                selected_indices = patched

            else:
                raise ValueError(
                    f"Unknown reject_mode '{reject_mode}'. "
                    "Choose 'ignore' or 'replace'.")

        # ── Sort and return ────────────────────────────────────────────────
        if return_coordinates:
            # Re-derive coordinates for the (possibly patched) index list
            idx_in_arr = [np.where(ch_arr == ch)[0][0]
                          for ch in selected_indices
                          if len(np.where(ch_arr == ch)[0]) > 0]
            sel_ch_final = [int(ch_arr[i]) for i in idx_in_arr]
            sel_x_final  = [float(x_arr[i]) for i in idx_in_arr]
            sel_y_final  = [float(y_arr[i]) for i in idx_in_arr]
            sort_order   = np.lexsort((sel_x_final, sel_y_final))
            sel_ch_final = [sel_ch_final[i] for i in sort_order]
            sel_x_final  = [sel_x_final[i]  for i in sort_order]
            sel_y_final  = [sel_y_final[i]  for i in sort_order]
            return sel_ch_final, {'x': sel_x_final, 'y': sel_y_final}

        selected_indices.sort()
        return selected_indices

    
    def load_processed_data(self, probe_key: str,
                           selected_channels: Optional[List[int]] = None,
                           coord_dict: Optional[Dict] = None,
                           depth_average: bool = False,
                           downsample_rate: Optional[float] = None,
                           low_cut: Optional[float] = None,
                           high_cut: Optional[float] = None,
                           filter_order: int = 3,
                           preprocessing_method: str = 'decimate',
                           notch_freq: Optional[float] = None,
                           save_processed: bool = True,
                           force_reload: bool = False,
                           overwrite_warning: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and optionally process data from a probe.

        Args:
            probe_key: Probe identifier (e.g., 'A_AP', 'B_AP', 'B_LFP')
            selected_channels: List of channel indices to load (None = all channels)
            coord_dict: Coordinate dictionary {'x': [...], 'y': [...]} for the
                selected channels, as returned by
                ``select_channels_by_coordinates(..., return_coordinates=True)[1]``.
                Required when ``depth_average=True``.
            depth_average: If True, average the LFP signals of all electrodes
                that share the same (shank, y-depth) within the selection.
                The result is one signal per unique (shank, y) pair, ordered by
                ascending depth.  ``coord_dict`` must be provided.
                Shank assignment is taken from the probe's shank_ids; if
                shank information is unavailable all channels are treated as
                belonging to a single shank.
            downsample_rate: Target sampling rate in Hz (None = no downsampling)
            low_cut: Low cutoff frequency for bandpass filter in Hz.
                Only used with preprocessing_method='resample_poly'.
            high_cut: High cutoff frequency for bandpass filter in Hz.
                Only used with preprocessing_method='resample_poly'.
            filter_order: Butterworth filter order for preprocessing_method='resample_poly'
                (default: 3).
            preprocessing_method: Downsampling pipeline to use. Options:
                - 'decimate' (default): 8th-order Chebyshev Type I anti-aliasing filter
                  followed by scipy.signal.decimate, replicating the method described
                  in the literature.  Recommended for wide-band AP data (e.g. 20 kHz
                  → 1250 Hz).  No bandpass pre-filtering is applied; the Chebyshev
                  filter acts as the anti-aliasing lowpass.
                - 'resample_poly': Original approach — optional Butterworth bandpass
                  (controlled by low_cut / high_cut) then polyphase resampling via
                  scipy.signal.resample_poly.  Suitable when a non-integer decimation
                  ratio is needed or when you want explicit passband control before
                  resampling.
            notch_freq: Optional line-noise frequency to notch-filter (Hz) before
                downsampling.  Applied by both preprocessing methods.
            save_processed: Whether to save processed data (default: True)
            force_reload: Force reload raw data even if processed exists (default: False)
            overwrite_warning: Show warning when overwriting existing files (default: True)

        Returns:
            data: Processed data with shape (channels, samples).
                  If depth_average=True this is (n_unique_depth_per_shank, samples).
            timestamps: Processed timestamps (subsampled to match output rate)
            coord_dict_out: Coordinate dictionary for the returned data rows.
                  Always present as third return value.
                  - If depth_average=False: identical to the input coord_dict
                    (or None if coord_dict was not supplied).
                  - If depth_average=True: updated dict reflecting the averaged
                    layout, with keys 'x', 'y', 'shank' — one entry per output
                    row, sorted by ascending (shank, depth).
        """
        probe_data = self.probe_info['probes'][probe_key]
        raw_sr = probe_data['sample_rate']
        
        # Create processed data filename
        processed_name = f"{probe_key}_processed"
        if selected_channels is not None:
            processed_name += f"_ch{len(selected_channels)}"
        if downsample_rate is not None:
            processed_name += f"_ds{int(downsample_rate)}_{preprocessing_method}"
        if low_cut is not None and high_cut is not None:
            processed_name += f"_bp{low_cut}-{high_cut}"
        
        # Path for processed data
        data_folder = self.get_probe_data_folder(probe_key)
        processed_file = data_folder / f"{processed_name}.npy"
        timestamps_file = data_folder / f"{processed_name}_timestamps.npy"
        
        # Check if processed data exists
        if not force_reload and processed_file.exists() and timestamps_file.exists():
            print(f"Loading processed data from {processed_file}")
            data = np.load(processed_file)
            timestamps = np.load(timestamps_file)
            return data, timestamps, coord_dict
        
        # Load raw data
        print(f"Loading raw data for probe {probe_key}...")
        raw_data, raw_timestamps = self.load_raw_data(probe_key, method='efficient_memory')
        
        # Select channels if specified
        if selected_channels is not None:
            raw_data = raw_data[selected_channels, :]
            print(f"Selected {len(selected_channels)} channels")

        # Depth-average: collapse channels sharing (shank, y) into one signal
        coord_dict_out = coord_dict  # default: pass through unchanged
        if depth_average:
            if coord_dict is None:
                raise ValueError(
                    "depth_average=True requires coord_dict. Pass the "
                    "coordinates dict returned by "
                    "select_channels_by_coordinates(..., return_coordinates=True)[1].")
            raw_data, coord_dict_out = self._depth_average_signals(
                probe_key, raw_data, selected_channels, coord_dict)
            print(f"Depth-averaged to {raw_data.shape[0]} unique-depth signals")

        # Downsample if specified
        if downsample_rate is not None and downsample_rate < raw_sr:
            print(f"Downsampling from {raw_sr} to {downsample_rate} Hz "
                  f"(method='{preprocessing_method}')...")
            data = filtering.preprocess_raw_signal(
                raw_data, raw_sr, downsample_rate,
                method=preprocessing_method,
                low_cut=low_cut,
                high_cut=high_cut,
                filter_order=filter_order,
                notch_freq=notch_freq,
            )
            # Build a matching timestamp array at the new sample rate
            decimation_factor = int(np.round(raw_sr / downsample_rate))
            timestamps = raw_timestamps[::decimation_factor][:data.shape[-1]].copy()
        else:
            # No downsampling: apply optional notch/bandpass and keep raw timestamps
            processed = raw_data
            if notch_freq is not None:
                processed = filtering.notch_filter(processed, raw_sr, notch_freq)
            if low_cut is not None and high_cut is not None:
                print(f"Applying bandpass filter {low_cut}-{high_cut} Hz...")
                processed = filtering.bandpass_filter(
                    processed, raw_sr, low_cut, high_cut, filter_order)
            data = processed
            timestamps = raw_timestamps
        
        # Save processed data if requested
        if save_processed:
            if processed_file.exists() and overwrite_warning:
                warnings.warn(f"Overwriting existing file: {processed_file}")
            print(f"Saving processed data to {processed_file}")
            np.save(processed_file, data)
            np.save(timestamps_file, timestamps)
        
        return data, timestamps, coord_dict_out
    
    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

    def _depth_average_signals(
        self,
        probe_key: str,
        raw_data: np.ndarray,
        selected_channels: Optional[List[int]],
        coord_dict: Dict,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Average LFP signals from electrodes sharing the same (shank, y-depth).

        Groups the rows of ``raw_data`` by unique (shank_id, y_position) pairs
        and replaces each group with the mean across its members.  The result
        is one row per unique (shank, y) pair, sorted by ascending depth within
        each shank.

        Parameters
        ----------
        probe_key : str
        raw_data : ndarray, shape (n_selected_channels, n_samples)
            Already channel-indexed data (i.e. after ``raw_data[selected_channels]``).
        selected_channels : list of int or None
            The channel numbers that index the rows of ``raw_data``.  If None
            the rows are treated as 0-based indices.
        coord_dict : dict  {'x': [...], 'y': [...]}
            Coordinates for each row of ``raw_data``, in the same order.

        Returns
        -------
        avg_data   : ndarray, shape (n_unique_depth_per_shank, n_samples)
        avg_coords : dict {'x': [...], 'y': [...], 'shank': [...]}
            Representative coordinates (mean x, same y, shank id) for each
            output row.
        """
        probe_data = self.probe_info['probes'][probe_key]
        all_ch     = np.array(probe_data['coordinates'].get(
                         'channels', list(range(probe_data['channel_count']))))
        shank_ids_all = np.array(
            probe_data.get('shanks', {}).get('shank_ids', [0] * len(all_ch)))

        x_arr = np.array(coord_dict['x'], dtype=float)
        y_arr = np.array(coord_dict['y'], dtype=float)

        n_rows = raw_data.shape[0]
        chs    = selected_channels if selected_channels is not None                  else list(range(n_rows))

        # Map each row → its shank id
        shank_of_row = []
        for ch in chs:
            idx = np.where(all_ch == ch)[0]
            shank_of_row.append(int(shank_ids_all[idx[0]]) if len(idx) > 0 else 0)
        shank_of_row = np.array(shank_of_row)

        # Group by (shank, y)  and sort
        groups: Dict[Tuple[int, float], List[int]] = {}
        for row_i, (shank, y) in enumerate(zip(shank_of_row, y_arr)):
            key = (int(shank), float(y))
            groups.setdefault(key, []).append(row_i)

        sorted_keys = sorted(groups.keys(), key=lambda k: (k[0], k[1]))

        avg_rows   = []
        avg_x_out  = []
        avg_y_out  = []
        avg_sh_out = []

        for (shank, y), row_indices in [(k, groups[k]) for k in sorted_keys]:
            rows = raw_data[row_indices, :]
            avg_rows.append(np.mean(rows, axis=0))
            avg_x_out.append(float(np.mean(x_arr[row_indices])))
            avg_y_out.append(y)
            avg_sh_out.append(shank)

            if len(row_indices) > 1:
                print(f"  Averaged {len(row_indices)} channels at "
                      f"shank={shank}, y={y:.0f} µm "
                      f"→ x_mean={avg_x_out[-1]:.1f} µm")

        avg_data   = np.vstack(avg_rows)
        avg_coords = {'x': avg_x_out, 'y': avg_y_out, 'shank': avg_sh_out}
        return avg_data, avg_coords

    def get_recording_info(self) -> Dict:
        """Return the recording structure information."""
        return self.recording_structure
    
    def list_available_probes(self) -> List[str]:
        """List all available probes."""
        return list(self.probe_info['probes'].keys())
    
    def get_probe_coordinates(self, probe_key: str) -> Dict:
        """Get electrode coordinates for a specific probe."""
        if probe_key not in self.probe_info['probes']:
            raise ValueError(f"Probe {probe_key} not found")
        return self.probe_info['probes'][probe_key].get('coordinates', {})
    
    def print_summary(self):
        """Print a summary of the loader configuration."""
        print("\n" + "="*60)
        print("NEUROPIXELS DATA LOADER SUMMARY")
        print("="*60)
        print(f"Folder: {self.folder}")
        print(f"Session: {self.session}")
        print(f"Base Path: {self.base_data_path}")
        print(f"\nRecording Structure:")
        print(f"  Record Node: {self.recording_structure['record_node']}")
        print(f"  Experiment: {self.recording_structure['experiment']}")
        print(f"  Recording: {self.recording_structure['recording']}")
        print(f"  Continuous Path: {self.recording_structure['continuous_path']}")
        
        print(f"\nProbes Found:")
        for probe_key, probe_data in self.probe_info['probes'].items():
            print(f"  - {probe_key}:")
            print(f"    Type: {probe_data['probe_type']}")
            print(f"    Sample Rate: {probe_data['sample_rate']} Hz")
            print(f"    Channels: {probe_data['channel_count']}")
            if probe_data['coordinates'].get('x'):
                print(f"    Coordinates: {len(probe_data['coordinates']['x'])} electrodes")
            print(f"    Folder: {probe_data['folder_name']}")
        print("="*60)


def get_settings_path(folder, session):
    
    raw_session_path = f"{data_path}/{folder}/{folder}_{session}/{folder}_{session}_recording/"
    data_pattern = os.path.join(raw_session_path, "Record Node */")
    # Define path to the settings XML file
    xml_path = glob.glob(data_pattern)[0]
    xml_file = 'settings.xml'

    xml_file = f'{xml_path}/{xml_file}'  # Update with your actual path    

    return xml_path, xml_file

    
def parse_neuropixels_settings(xml_file_path, summary=False):
    """
    Parse Neuropixels settings from XML file and return structured probe information.
    
    Args:
        xml_file_path (str): Path to the settings.xml file
        summary (bool): Whether to print a summary
        
    Returns:
        dict: Structured dictionary with all probe information
    """
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    
    # Get recording path from CONTROL PANEL
    control_panel = root.find('.//CONTROLPANEL')
    record_path = control_panel.get('recordPath', '') if control_panel is not None else ''
    
    # Initialize result structure
    result = {
        'record_path': record_path,
        'probes': {},
        'folders_to_create': []
    }
    
    # Find all Neuropix-PXI processors
    neuropix_processors = root.findall(".//PROCESSOR[@pluginName='Neuropix-PXI']")
    
    for processor in neuropix_processors:
        processor_name = processor.get('name')
        node_id = processor.get('nodeId')
        
        # Extract stream information
        streams = processor.findall('.//STREAM')
        
        # Create probe entries from streams
        for stream in streams:
            stream_name = stream.get('name')
            sample_rate = float(stream.get('sample_rate', '0'))
            channel_count = int(stream.get('channel_count', '0'))
            device_name = stream.get('device_name', '')
            
            # Create probe key using processor name and stream name
            probe_key = f'{processor_name}-{node_id}.{stream_name}'
            
            # Store basic stream info
            result['probes'][probe_key] = {
                'processor_name': processor_name,
                'node_id': node_id,
                'stream_name': stream_name,
                'sample_rate': sample_rate,
                'channel_count': channel_count,
                'device_name': device_name,
                'folder_name': probe_key,
                'coordinates': {'x': [], 'y': [], 'channels': []},
                'shanks': {'shank_ids': [], 'shank_mapping': {}}
            }
            
            # Add to folders to create list
            result['folders_to_create'].append(probe_key)
        
        # Extract coordinates from NP_PROBE elements and match by device_name
        np_probes = processor.findall('.//NP_PROBE')
        
        for np_probe in np_probes:
            # Extract coordinates and shank info
            coords, shank_info = extract_electrode_coordinates_with_shanks(np_probe)
            
            # Get the probe name from NP_PROBE (this should match device_name in streams)
            probe_name = np_probe.get('probe_name', '')
            
            # Find matching stream by device_name
            for probe_key, probe_data in result['probes'].items():
                if probe_data['device_name'] == probe_name:
                    # Assign coordinates to this stream
                    probe_data['coordinates'] = coords
                    probe_data['shanks'] = shank_info
                    probe_data['electrode_count'] = len(coords.get('x', []))
                    probe_data['num_shanks'] = len(set(shank_info.get('shank_ids', []))) if shank_info.get('shank_ids') else 1
                    probe_data['electrode_config'] = np_probe.get('electrodeConfigurationPreset', '')
                    probe_data['probe_serial'] = np_probe.get('probe_serial_number', '')
                    probe_data['probe_name'] = probe_name
    
    # Remove duplicates from folders_to_create
    result['folders_to_create'] = list(set(result['folders_to_create']))
    
    # Print summary if requested
    if summary:
        print_probe_summary(result)
    
    return result


def extract_electrode_coordinates_with_shanks(np_probe_element):
    """
    Extract electrode X and Y coordinates and shank information from NP_PROBE element.
    
    The CHANNELS attribute contains shank information in format "electrode:shank"
    e.g., "0:0" means electrode 0, shank 0; "48:1" means electrode 48, shank 1
    """
    coordinates = {'x': [], 'y': [], 'channels': []}
    shank_info = {'shank_ids': [], 'shank_mapping': {}}
    
    # Find ELECTRODE_XPOS and ELECTRODE_YPOS elements
    xpos_elem = np_probe_element.find('.//ELECTRODE_XPOS')
    ypos_elem = np_probe_element.find('.//ELECTRODE_YPOS')
    channels_elem = np_probe_element.find('.//CHANNELS')
    
    # Extract coordinates from XPOS and YPOS
    if xpos_elem is not None and ypos_elem is not None:
        # Collect all channel numbers and x coordinates
        x_dict = {}
        for key, value in xpos_elem.attrib.items():
            if key.startswith('CH'):
                channel_num = int(key[2:])
                x_dict[channel_num] = int(value)
        
        y_dict = {}
        for key, value in ypos_elem.attrib.items():
            if key.startswith('CH'):
                channel_num = int(key[2:])
                y_dict[channel_num] = int(value)
        
        # Get all channel numbers that have both x and y
        all_channels = sorted(set(x_dict.keys()) & set(y_dict.keys()))
        
        coordinates['channels'] = all_channels
        coordinates['x'] = [x_dict[ch] for ch in all_channels]
        coordinates['y'] = [y_dict[ch] for ch in all_channels]
    
    # Extract shank information from CHANNELS element
    if channels_elem is not None:
        for key, value in channels_elem.attrib.items():
            if key.startswith('CH'):
                try:
                    channel_num = int(key[2:])
                    # Parse the value which should be in format "electrode:shank"
                    if ':' in value:
                        electrode, shank = value.split(':')
                        shank_id = int(shank)
                        shank_info['shank_mapping'][channel_num] = shank_id
                    else:
                        # If no colon, assume shank 0
                        shank_info['shank_mapping'][channel_num] = 0
                except (ValueError, IndexError):
                    pass
    
    # Build shank_ids list in same order as coordinates['channels']
    shank_info['shank_ids'] = [shank_info['shank_mapping'].get(ch, 0) for ch in coordinates['channels']]
    
    return coordinates, shank_info


def print_probe_summary(probe_info):
    """Print a formatted summary of the probe information."""
    print("\n" + "="*80)
    print("NEUROPIXELS PROBE CONFIGURATION SUMMARY")
    print("="*80)
    print(f"Record Path: {probe_info['record_path']}")
    print(f"Number of probes: {len(probe_info['probes'])}")
    print("\nProbe Details:")
    print("-"*80)
    
    for probe_key, probe_data in probe_info['probes'].items():
        print(f"\n {probe_key}")
        print(f"   Stream: {probe_data['stream_name']}")
        print(f"   Device: {probe_data['device_name']}")
        print(f"   Sample Rate: {probe_data['sample_rate']} Hz")
        print(f"   Channel Count: {probe_data['channel_count']}")
        
        num_shanks = probe_data.get('num_shanks', 1)
        print(f"   Number of Shanks: {num_shanks}")
        
        coords = probe_data.get('coordinates', {})
        if coords.get('x'):
            print(f"   Electrodes with coordinates: {len(coords['x'])}")
            unique_x = np.unique(coords['x'])
            print(f"   Unique X positions: {unique_x}")
            print(f"   Y-range: {min(coords['y'])} to {max(coords['y'])} μm")
            
            # Show shank distribution
            shank_ids = probe_data.get('shanks', {}).get('shank_ids', [])
            if shank_ids and len(set(shank_ids)) > 1:
                print(f"\n   Shank Distribution:")
                for shank_id in sorted(set(shank_ids)):
                    count = shank_ids.count(shank_id)
                    print(f"     Shank {shank_id}: {count} channels")
            
            # Show first few channels
            print(f"\n   First 10 channels (sorted by depth):")
            print("   Channel | X (μm) | Y (μm) | Shank")
            print("   " + "-"*40)
            
            # Sort by y-coordinate
            sort_idx = np.argsort(coords['y'])
            n_display = min(10, len(coords['channels']))
            for i in range(n_display):
                idx = sort_idx[i]
                ch = coords['channels'][idx]
                x = coords['x'][idx]
                y = coords['y'][idx]
                shank = shank_ids[idx] if shank_ids else 0
                print(f"   {ch:7d} | {x:5d} | {y:5d} | {shank}")
        
        print(f"   Folder name: {probe_data['folder_name']}")
    
    print("\nFolders to create:")
    for folder in probe_info['folders_to_create']:
        print(f"   - {folder}")
    print("="*80)



def create_kilosort_chanmap(probe_info, probe_key):
    """
    Create Kilosort channel map structure for a probe.
    
    Args:
        probe_info: Dictionary from parse_neuropixels_settings
        probe_key: Key identifying the probe
    
    Returns:
        Dictionary with channel map structure for Kilosort
    """
    # Get probe data
    probe = probe_info['probes'][probe_key]
    coords = probe.get('coordinates', {})
    shanks = probe.get('shanks', {})
    
    if not coords.get('x') or not coords.get('y'):
        raise ValueError(f"No coordinate information available for probe {probe_key}")
    
    # Get coordinates and shank assignments
    xcoords = np.array(coords['x'])
    ycoords = np.array(coords['y'])
    n_channels = len(xcoords)
    
    # Get shank IDs (default to 0 if not available)
    if shanks.get('shank_ids'):
        shank_ids = np.array(shanks['shank_ids'])
    else:
        shank_ids = np.zeros(n_channels, dtype=int)
    
    # Sort by y-coordinate (depth) for proper ordering
    sort_idx = np.argsort(ycoords)
    xcoords_sorted = xcoords[sort_idx]
    ycoords_sorted = ycoords[sort_idx]
    shank_ids_sorted = shank_ids[sort_idx]
    
    # Create channel map structures
    chanMap = np.arange(1, n_channels + 1, dtype=np.float64).reshape(-1, 1)
    chanMap0ind = np.arange(0, n_channels, dtype=np.float64).reshape(-1, 1)
    connected = np.ones(n_channels, dtype=bool).reshape(-1, 1)
    
    # Use shank IDs for kcoords (Kilosort uses this for grouping)
    kcoords = shank_ids_sorted.astype(np.float64).reshape(-1, 1)
    
    # Sort everything by depth
    chanMap_sorted = chanMap[sort_idx]
    chanMap0ind_sorted = chanMap0ind[sort_idx]
    connected_sorted = connected[sort_idx]
    
    # Create name for the channel map
    device_name = probe['device_name']
    
    # Extract probe identifier from the key
    if 'ProbeA' in probe_key:
        identifier = 'ProbeA'
    elif 'ProbeB-AP' in probe_key:
        identifier = 'ProbeB_AP'
    elif 'ProbeB-LFP' in probe_key:
        identifier = 'ProbeB_LFP'
    else:
        identifier = probe_key.split('.')[-1] if '.' in probe_key else probe_key
    
    if 'Neuropixels 2.0' in device_name:
        base_name = f"NP2_kilosortChanMap"
    elif 'Neuropixels 1.0' in device_name:
        base_name = f"NP1_kilosortChanMap"
    else:
        base_name = f"kilosortChanMap"
    
    name = f"{base_name}_{identifier}"
    
    # Create the struct
    chanmap_struct = {
        'chanMap': chanMap_sorted,
        'chanMap0ind': chanMap0ind_sorted,
        'connected': connected_sorted,
        'kcoords': kcoords,
        'xcoords': xcoords_sorted.astype(np.float64).reshape(-1, 1),
        'ycoords': ycoords_sorted.astype(np.float64).reshape(-1, 1),
        'name': name
    }
    
    return chanmap_struct


def save_probe_chanmaps(probe_info, output_dir=None, output_prefix="chanMap"):
    """
    Save channel maps for all probes to separate .mat files.
    
    Args:
        probe_info: Dictionary from parse_neuropixels_settings
        output_dir: Directory to save files (default: current directory)
        output_prefix: Prefix for output filenames
        
    Returns:
        List of saved file paths
    """
    from scipy.io import savemat
    import os
    import numpy as np
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    saved_files = []
    
    for probe_key, probe_data in probe_info['probes'].items():
        print(f"\n{'='*80}")
        print(f"Processing Probe: {probe_key}")
        print(f"{'='*80}")
        print(f"  Stream: {probe_data.get('stream_name', 'N/A')}")
        print(f"  Device: {probe_data['device_name']}")
        print(f"  Sample Rate: {probe_data['sample_rate']} Hz")
        print(f"  Channels: {probe_data['channel_count']}")
        
        # Check if coordinates are available
        coords = probe_data.get('coordinates', {})
        if not coords.get('x') or not coords.get('y'):
            print(f"  WARNING: No coordinate information available for {probe_key}")
            print(f"  Skipping this probe...")
            continue
        
        shanks = probe_data.get('shanks', {})
        num_shanks = probe_data.get('num_shanks', 1)
        
        xcoords = np.array(coords['x'])
        ycoords = np.array(coords['y'])
        print(f"  X-range: {xcoords.min():.0f} to {xcoords.max():.0f} μm")
        print(f"  Y-range: {ycoords.min():.0f} to {ycoords.max():.0f} μm")
        print(f"  Unique X positions: {np.unique(xcoords)}")
        print(f"  Number of shanks: {num_shanks}")
        
        # Create channel map
        try:
            chanmap = create_kilosort_chanmap(probe_info, probe_key)
            
            # Generate filename based on probe key
            if 'ProbeA' in probe_key:
                identifier = 'ProbeA'
            elif 'ProbeB-AP' in probe_key:
                identifier = 'ProbeB_AP'
            elif 'ProbeB-LFP' in probe_key:
                identifier = 'ProbeB_LFP'
            else:
                identifier = probe_key.split('.')[-1] if '.' in probe_key else probe_key
            
            if 'Neuropixels 2.0' in probe_data['device_name']:
                filename = f"NP2_kilosortChanMap_{identifier}.mat"
            elif 'Neuropixels 1.0' in probe_data['device_name']:
                filename = f"NP1_kilosortChanMap_{identifier}.mat"
            else:
                filename = f"{output_prefix}_{identifier}.mat"
            
            if output_dir:
                filepath = os.path.join(output_dir, filename)
            else:
                filepath = filename
            
            # Save to .mat file
            savemat(filepath, chanmap, format='5')
            saved_files.append(filepath)
            print(f"    Saved to: {filepath}")
            
            # Print first few channels as sample
            print("\n  First 10 channels (sorted by depth):")
            print("  Index | chanMap | chanMap0ind | kcoords | xcoords | ycoords")
            print("  " + "-"*70)
            n_display = min(10, len(chanmap['chanMap']))
            for j in range(n_display):
                print(f"  {j+1:5d} | {int(chanmap['chanMap'][j][0]):7d} | "
                      f"{int(chanmap['chanMap0ind'][j][0]):11d} | "
                      f"{int(chanmap['kcoords'][j][0]):7d} | "
                      f"{int(chanmap['xcoords'][j][0]):7d} | "
                      f"{int(chanmap['ycoords'][j][0]):7d}")
            
            # Print shank distribution
            unique_shanks = np.unique(chanmap['kcoords'])
            print(f"\n  Shank distribution:")
            for shank in unique_shanks:
                count = np.sum(chanmap['kcoords'] == shank)
                print(f"    Shank {int(shank)}: {count} channels")
            
            # Print statistics
            print(f"\n  Statistics:")
            print(f"    Channel depth range: {chanmap['ycoords'].min():.0f} to {chanmap['ycoords'].max():.0f} μm")
            print(f"    Number of shanks/groups: {len(unique_shanks)}")
            
        except Exception as e:
            print(f"   Error creating channel map for {probe_key}: {e}")
    
    print(f"\n{'='*80}")
    print(f"Summary: Saved {len(saved_files)} channel map files")
    if saved_files:
        print("Files:")
        for f in saved_files:
            print(f"  - {f}")
    print(f"{'='*80}")
    
    return saved_files