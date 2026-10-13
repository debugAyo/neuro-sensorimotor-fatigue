import numpy as np
from scipy.signal import butter, filtfilt, welch

def generate_synthetic_emg(duration=20, fs=1000):
    """
    Generates a synthetic sEMG signal that simulates progressive muscle fatigue.
    Fatigue is modeled by shifting the median frequency downward and increasing RMS amplitude.
    """
    t = np.arange(0, duration, 1/fs)
    num_samples = len(t)
    raw_signal = np.zeros(num_samples)
    
    # Simulate signal sliding from fresh (center ~150Hz) to fatigued (center ~60Hz)
    for i in range(num_samples):
        progress = i / num_samples
        center_freq = 150 - (90 * progress) 
        bandwidth = 40 - (15 * progress)
        amplitude = 1.0 + (1.5 * progress)
        
        noise = np.random.normal(0, 1)
        raw_signal[i] = amplitude * noise * np.sin(2 * np.pi * center_freq * t[i])

    # Add 50Hz powerline interference and broadband measurement noise
    powerline_noise = 0.3 * np.sin(2 * np.pi * 50 * t)
    measurement_noise = np.random.normal(0, 0.2, num_samples)
    
    return t, raw_signal + powerline_noise + measurement_noise

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Applies a Butterworth bandpass filter to remove motion artifacts and high-freq noise."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def butter_notch_filter(data, notch_freq, fs, q=30):
    """Applies a notch filter to attenuate powerline interference (50Hz/60Hz)."""
    nyq = 0.5 * fs
    w0 = notch_freq / nyq
    b, a = butter(2, [w0 - w0/q, w0 + w0/q], btype='bandstop')
    return filtfilt(b, a, data)

def calculate_median_frequency(segment, fs):
    """Computes the Median Frequency (MDF) of an sEMG window using Welch's PSD."""
    frequencies, psd = welch(segment, fs=fs, nperseg=len(segment))
    cumulative_power = np.cumsum(psd)
    total_power = cumulative_power[-1]
    
    # Find the first index where cumulative power crosses 50%
    mdf_idx = np.where(cumulative_power >= total_power / 2)
    
    # FIX: Return the actual frequency value corresponding to that index
    if len(mdf_idx) > 0 and len(mdf_idx[0]) > 0:
        return frequencies[mdf_idx[0][0]]
    return 0.0


def calculate_rms(segment):
    """Computes the Root Mean Square (RMS) amplitude of a segment."""
    return np.sqrt(np.mean(segment**2))

def process_emg_pipeline(raw_signal, fs, window_size_sec=1.0, overlap_pct=0.5):
    """Runs sliding window analysis to extract real-time feature trajectories."""
    step_size = int(window_size_sec * fs * (1 - overlap_pct))
    window_samples = int(window_size_sec * fs)
    
    # Pre-process signals: Bandpass (20-450Hz) and Notch (50Hz)
    filtered_signal = butter_bandpass_filter(raw_signal, 20, 450, fs)
    filtered_signal = butter_notch_filter(filtered_signal, 50, fs)
    
    mdf_trajectory = []
    rms_trajectory = []
    time_vectors = []
    
    # Sliding window execution
    for start in range(0, len(filtered_signal) - window_samples, step_size):
        end = start + window_samples
        segment = filtered_signal[start:end]
        
        mdf = calculate_median_frequency(segment, fs)
        rms = calculate_rms(segment)
        
        mdf_trajectory.append(mdf)
        rms_trajectory.append(rms)
        time_vectors.append((start + end) / (2 * fs))
        
    return np.array(time_vectors), np.array(mdf_trajectory), np.array(rms_trajectory), filtered_signal
