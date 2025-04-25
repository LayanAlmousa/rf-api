from scipy.signal import butter, filtfilt, detrend, find_peaks, peak_widths
from scipy.stats import entropy as sp_entropy
from antropy import sample_entropy as ap_sampen
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

def preprocess_gsr_dataset(data: pd.DataFrame, fs: float) -> pd.DataFrame:
    """
    Decomposes each GSR segment into tonic and phasic components.
    """
    from scipy.signal import butter, filtfilt, detrend

    nyq  = fs / 2.0
    b_t, a_t = butter(2, 0.05/nyq, btype='low')
    b_p, a_p = butter(2, [0.5/nyq, 3.0/nyq], btype='band')

    out = data.copy()

    def decompose(seg):
        x = np.ravel(seg.values if isinstance(seg, (pd.Series, pd.DataFrame)) else seg)
        x = x - x.mean()
        tonic = filtfilt(b_t, a_t, x)
        phasic = filtfilt(b_p, a_p, x - tonic)
        phasic = detrend(phasic)
        tonic_wr = pd.Series(tonic)
        phasic_wr = pd.Series(phasic)
        return tonic_wr, phasic_wr

    # FINAL FIX:
    out[['Tonic_Data', 'Phasic_Data']] = pd.DataFrame(out['GSR_Data'].apply(decompose).tolist(), index=out.index)

    return out


def segment_gsr_data(clean_data: pd.DataFrame,
                     fs: float,
                     window_sec: float = 10.0,
                     overlap_sec: float = 5.0) -> pd.DataFrame:
    """
    Splits each segment into overlapping windows and returns
    three‐channel data (raw, tonic, phasic) plus label.
    """
    window_samples = int(window_sec * fs)
    step_samples   = int((window_sec - overlap_sec) * fs)

    rows = []
    for _, row in clean_data.iterrows():
        x_raw     = np.ravel(row['GSR_Data'])
        x_tonic   = np.ravel(row['Tonic_Data'])
        x_phasic  = np.ravel(row['Phasic_Data'])
        label     = row['Stress'] if 'Stress' in row else 'unknown'
        n         = len(x_raw)
        start     = 0

        while start < n:
            end = start + window_samples

            if end <= n:
                seg_raw    = x_raw[start:end]
                seg_tonic  = x_tonic[start:end]
                seg_phasic = x_phasic[start:end]
            else:
                seg_raw    = x_raw[n-window_samples:n]
                seg_tonic  = x_tonic[n-window_samples:n]
                seg_phasic = x_phasic[n-window_samples:n]
                start = n

            rows.append({
                'Raw':    seg_raw,
                'Tonic':  seg_tonic,
                'Phasic': seg_phasic,
                'Stress': label
            })
            start += step_samples

    return pd.DataFrame(rows)



def extract_features_matrix_optimized(segmented_data: pd.DataFrame,
                                      fs: float,
                                      n_jobs: int = -1) -> pd.DataFrame:
    """
    Fast feature extraction across Raw/Tonic/Phasic channels.

    Returns a DataFrame with these columns per window:
      • Stress (0/1)
      • Raw_mean, Raw_std, Raw_range, Raw_auc
      • Tonic_mean, Tonic_std, Tonic_range, Tonic_auc, Tonic_slope
      • Phasic_mean, Phasic_std, Phasic_range, Phasic_auc
      • Phasic_bp_0.05_0.10, Phasic_bp_0.10_0.30, Phasic_bp_0.30_0.50
      • Phasic_spec_entropy
      • Phasic_sampen
      • Phasic_n_peaks, Phasic_mean_peak_amp, Phasic_mean_peak_width
    """
    # Stack arrays
    raw   = np.stack(segmented_data['Raw'].values)
    tonic = np.stack(segmented_data['Tonic'].values)
    phasic= np.stack(segmented_data['Phasic'].values)
    labels= (segmented_data['Stress']=='yes').astype(int).values

    n_windows, N = raw.shape

    # Precompute time vector & denom for slope
    t = np.arange(N)/fs
    t_mean = t.mean()
    denom = ((t - t_mean)**2).sum()

    # 1) Time‑domain & AUC vectorized
    def time_features(x):
        mean = x.mean(axis=1)
        std  = x.std(axis=1)
        rng  = np.ptp(x, axis=1)  # use the top‐level function instead
        auc  = np.trapz(np.abs(x), dx=1/fs, axis=1)
        return mean, std, rng, auc


    rm, rs, rr, ra = time_features(raw)
    tm, ts, tr, ta = time_features(tonic)
    pm, ps, pr, pa = time_features(phasic)

    # 2) Tonic slope vectorized
    tonic_centered = tonic - tm[:,None]
    slope = (tonic_centered * (t - t_mean)).sum(axis=1) / denom

    # 3) Phasic spectral features vectorized
    freqs      = np.fft.rfftfreq(N, 1/fs)
    fft_ph     = np.fft.rfft(phasic, axis=1)
    psd_ph     = (np.abs(fft_ph)**2) / N

    bp_05_10 = np.trapz(psd_ph[:, (freqs>=0.05)&(freqs<0.10)],
                       freqs[(freqs>=0.05)&(freqs<0.10)], axis=1)
    bp_10_30 = np.trapz(psd_ph[:, (freqs>=0.10)&(freqs<0.30)],
                       freqs[(freqs>=0.10)&(freqs<0.30)], axis=1)
    bp_30_50 = np.trapz(psd_ph[:, (freqs>=0.30)&(freqs<0.50)],
                       freqs[(freqs>=0.30)&(freqs<0.50)], axis=1)

    # Spectral entropy
    p_norm = psd_ph / psd_ph.sum(axis=1, keepdims=True)
    spec_ent = sp_entropy(p_norm, base=2, axis=1)

    # 4) Per‑window sample entropy + event features (parallel)
    def phasic_events(i):
        x = phasic[i]
        # fast sample entropy
        se = ap_sampen(x)
        # SCR peaks
        peaks, props = find_peaks(x,
                                  height=np.percentile(x,75),
                                  distance=int(0.5*fs))
        n_peaks = len(peaks)
        amps    = props.get('peak_heights', x[peaks])
        mp_amp  = amps.mean() if n_peaks>0 else 0.0
        widths  = peak_widths(x, peaks, rel_height=0.5)[0] / fs
        mp_wid  = widths.mean() if widths.size>0 else 0.0
        return se, n_peaks, mp_amp, mp_wid

    results = Parallel(n_jobs=n_jobs)(
        delayed(phasic_events)(i) for i in range(n_windows)
    )
    sampens, n_peaks, mean_amps, mean_widths = zip(*results)

    # Assemble into DataFrame
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
    return df
