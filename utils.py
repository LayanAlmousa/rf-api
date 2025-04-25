from scipy.signal import butter, filtfilt, detrend, find_peaks, peak_widths
from scipy.stats import entropy as sp_entropy
from antropy import sample_entropy as ap_sampen
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

def preprocess_gsr_signal(gsr_data: pd.DataFrame, fs: float) -> dict:
    """
    Decomposes a single GSR segment into tonic and phasic components.

    Args:
        gsr_data (pd.DataFrame): a single column (n,1) GSR DataFrame
        fs (float): sampling frequency

    Returns:
        dict: {'GSR_Data': ..., 'Tonic_Data': ..., 'Phasic_Data': ...}
    """
    logging.debug("Starting preprocess_gsr_signal function.")
    
    nyq = fs / 2.0
    b_t, a_t = butter(2, 0.05 / nyq, btype='low')
    b_p, a_p = butter(2, [0.5 / nyq, 3.0 / nyq], btype='band')

    logging.debug(f"GSR data before cleaning: {gsr_data.head()}")

    x = np.ravel(gsr_data.values)
    x = x - x.mean()

    tonic = filtfilt(b_t, a_t, x)
    phasic = filtfilt(b_p, a_p, x - tonic)
    phasic = detrend(phasic)

    logging.debug(f"Tonic and Phasic data after filtering.")
    logging.debug(f"Tonic data: {tonic[:5]}")
    logging.debug(f"Phasic data: {phasic[:5]}")

    return {
        'GSR_Data': pd.Series(x),
        'Tonic_Data': pd.Series(tonic),
        'Phasic_Data': pd.Series(phasic)
    }


def segment_single_gsr_segment(preprocessed: dict, fs: float,
                                window_sec: float = 10.0,
                                overlap_sec: float = 5.0) -> pd.DataFrame:
    """
    Segments a single preprocessed GSR signal into overlapping windows.

    Returns:
        pd.DataFrame: with columns ['Raw', 'Tonic', 'Phasic', 'Stress']
    """
    logging.debug("Starting segment_single_gsr_segment function.")

    window_samples = int(window_sec * fs)
    step_samples = int((window_sec - overlap_sec) * fs)

    x_raw = preprocessed['GSR_Data'].values
    x_tonic = preprocessed['Tonic_Data'].values
    x_phasic = preprocessed['Phasic_Data'].values

    logging.debug(f"Segmentation parameters: window_samples={window_samples}, step_samples={step_samples}")

    rows = []
    start = 0
    n = len(x_raw)

    while start < n:
        end = start + window_samples

        if end <= n:
            seg_raw = x_raw[start:end]
            seg_tonic = x_tonic[start:end]
            seg_phasic = x_phasic[start:end]
        else:
            seg_raw = x_raw[n - window_samples:n]
            seg_tonic = x_tonic[n - window_samples:n]
            seg_phasic = x_phasic[n - window_samples:n]
            start = n

        logging.debug(f"Segment {start // window_samples + 1}:")
        logging.debug(f"Raw segment: {seg_raw[:5]}")
        logging.debug(f"Tonic segment: {seg_tonic[:5]}")
        logging.debug(f"Phasic segment: {seg_phasic[:5]}")

        rows.append({
            'Raw': seg_raw,
            'Tonic': seg_tonic,
            'Phasic': seg_phasic,
            'Stress': 'unknown'
        })

        start += step_samples

    df_segmented = pd.DataFrame(rows)
    logging.debug(f"Segmentation complete. Number of segments: {len(df_segmented)}")
    return df_segmented


def extract_features_matrix_optimized(segmented_data: pd.DataFrame,
                                      fs: float,
                                      n_jobs: int = -1) -> pd.DataFrame:
    """
    Feature extraction for one segmented GSR session.
    """
    logging.debug("Starting feature extraction function.")

    raw   = np.stack(segmented_data['Raw'].values)
    tonic = np.stack(segmented_data['Tonic'].values)
    phasic= np.stack(segmented_data['Phasic'].values)
    labels= (segmented_data['Stress'] == 'yes').astype(int).values

    logging.debug(f"Shape of raw, tonic, and phasic data:")
    logging.debug(f"raw.shape = {raw.shape}, tonic.shape = {tonic.shape}, phasic.shape = {phasic.shape}")

    n_windows, N = raw.shape
    t = np.arange(N) / fs
    t_mean = t.mean()
    denom = ((t - t_mean) ** 2).sum()

    logging.debug(f"Computing time-domain features.")
    
    def time_features(x):
        mean = x.mean(axis=1)
        std  = x.std(axis=1)
        rng  = np.ptp(x, axis=1)
        auc  = np.trapz(np.abs(x), dx=1/fs, axis=1)
        return mean, std, rng, auc

    rm, rs, rr, ra = time_features(raw)
    tm, ts, tr, ta = time_features(tonic)
    pm, ps, pr, pa = time_features(phasic)

    tonic_centered = tonic - tm[:, None]
    slope = (tonic_centered * (t - t_mean)).sum(axis=1) / denom

    freqs = np.fft.rfftfreq(N, 1/fs)
    fft_ph = np.fft.rfft(phasic, axis=1)
    psd_ph = (np.abs(fft_ph) ** 2) / N

    bp_05_10 = np.trapz(psd_ph[:, (freqs >= 0.05) & (freqs < 0.10)], freqs[(freqs >= 0.05) & (freqs < 0.10)], axis=1)
    bp_10_30 = np.trapz(psd_ph[:, (freqs >= 0.10) & (freqs < 0.30)], freqs[(freqs >= 0.10) & (freqs < 0.30)], axis=1)
    bp_30_50 = np.trapz(psd_ph[:, (freqs >= 0.30) & (freqs < 0.50)], freqs[(freqs >= 0.30) & (freqs < 0.50)], axis=1)

    p_norm = psd_ph / psd_ph.sum(axis=1, keepdims=True)
    spec_ent = sp_entropy(p_norm, base=2, axis=1)

    logging.debug(f"Extracting per-window sample entropy and event features.")
    def phasic_events(i):
        x = phasic[i]
        se = ap_sampen(x)
        peaks, props = find_peaks(x,
                                  height=np.percentile(x, 75),
                                  distance=int(0.5 * fs))
        n_peaks = len(peaks)
        amps = props.get('peak_heights', x[peaks])
        mp_amp = amps.mean() if n_peaks > 0 else 0.0
        widths = peak_widths(x, peaks, rel_height=0.5)[0] / fs
        mp_wid = widths.mean() if widths.size > 0 else 0.0
        return se, n_peaks, mp_amp, mp_wid

    results = Parallel(n_jobs=n_jobs)(delayed(phasic_events)(i) for i in range(n_windows))
    sampens, n_peaks, mean_amps, mean_widths = zip(*results)

    df = pd.DataFrame({
        'Stress': labels,
        'Raw_mean': rm, 'Raw_std': rs, 'Raw_range': rr, 'Raw_auc': ra,
        'Tonic_mean': tm, 'Tonic_std': ts, 'Tonic_range': tr, 'Tonic_auc': ta,
        'Tonic_slope': slope,
        'Phasic_mean': pm, 'Phasic_std': ps, 'Phasic_range': pr, 'Phasic_auc': pa,
        'Phasic_bp_0.05_0.10': bp_05_10,
        'Phasic_bp_0.10_0.30': bp_10_30,
        'Phasic_bp_0.30_0.50': bp_30_50,
        'Phasic_spec_entropy': spec_ent,
        'Phasic_sampen': sampens,
        'Phasic_n_peaks': n_peaks,
        'Phasic_mean_peak_amp': mean_amps,
        'Phasic_mean_peak_width': mean_widths
    })

    logging.debug(f"Feature extraction complete. Extracted {len(df)} features.")
    return df
