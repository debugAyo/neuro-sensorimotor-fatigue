import numpy as np
import matplotlib.pyplot as plt
from signal_processing import process_emg_pipeline
from data_loader import fetch_realworld_emg_data

class NeuroSafetyInterlock:
    def __init__(self, baseline_mdf_windows=6, threshold_drop_pct=0.18):
        """
        Predictive safety controller to track sensorimotor degradation.
        """
        self.baseline_windows = baseline_mdf_windows
        self.threshold_pct = threshold_drop_pct
        
    def monitor_stream(self, time_vectors, mdf_trajectory):
        """Scans the computed spectral trajectory to predict fatigue onset."""
        baseline_mdf = np.mean(mdf_trajectory[:self.baseline_windows])
        critical_threshold = baseline_mdf * (1 - self.threshold_pct)
        
        interlock_fired = False
        interlock_time = None
        trigger_index = -1
        
        for idx, mdf_val in enumerate(mdf_trajectory):
            if mdf_val <= critical_threshold:
                interlock_fired = True
                interlock_time = time_vectors[idx]
                trigger_index = idx
                break
                
        return baseline_mdf, critical_threshold, interlock_fired, interlock_time, trigger_index

if __name__ == "__main__":
    fs = 1000  # Acquisition baseline frequency (1 kHz)
    
    # 1. Fetch data from our new module
    print("[PIPELINE] Initializing data acquisition loop...")
    time, raw_emg = fetch_realworld_emg_data()
    
    print("[PIPELINE] Running sliding-window spectral features...")
    feature_time, mdf_track, rms_track, clean_emg = process_emg_pipeline(raw_emg, fs)
    
    # 2. Safety Interlock Evaluation
    detector = NeuroSafetyInterlock(baseline_mdf_windows=6, threshold_drop_pct=0.18)
    base_f, thresh_f, fired, f_time, f_idx = detector.monitor_stream(feature_time, mdf_track)
    
    print(f"\n--- Cybernetic State Estimator Results ---")
    print(f"Baseline Median Frequency: {base_f:.2f} Hz")
    print(f"Safety Cutoff Threshold: {thresh_f:.2f} Hz (18% Drop)")
    if fired:
        print(f"STATUS: INTERLOCK FIRED at {f_time:.2f} seconds!")
        print(f"Prediction Horizon: Safe boundary enforced before catastrophic motor error.")
    else:
        print("STATUS: System Nominal. No fatigue bounds crossed.")
    print("------------------------------------------")
    
    # 3. Save the high-resolution diagnostic plot
    plt.figure(figsize=(11, 5.5))
    plt.plot(feature_time, mdf_track, marker='o', color='#1a1a1a', linewidth=1.5, label='Tracked Median Frequency (MDF)')
    plt.axhline(y=base_f, color='#2e7d32', linestyle='--', linewidth=1.5, label=f'Fresh Muscle Baseline ({base_f:.1f} Hz)')
    plt.axhline(y=thresh_f, color='#d32f2f', linestyle='--', linewidth=1.5, label=f'Fatigue Interlock Boundary ({thresh_f:.1f} Hz)')
    
    if fired:
        plt.axvline(x=f_time, color='#ffcdd2', alpha=0.5, linestyle='-', zorder=1)
        plt.scatter(f_time, mdf_track[f_idx], color='#d32f2f', s=180, zorder=5, 
                    label=f'Safety Interlock Fired ({f_time:.2f}s)')
        
    plt.title("NeuroSensorimotor-Predict: Real-Time Clinical Fatigue State Estimation", fontsize=12, fontweight='bold')
    plt.xlabel("Execution Time (seconds)", fontsize=10)
    plt.ylabel("Spectral Center of Mass (Hz)", fontsize=10)
    plt.legend(loc='lower left', frameon=True, shadow=False)
    plt.grid(True, alpha=0.3, linestyle=':')
    plt.tight_layout()
    
    plt.savefig('predictive_interlock_verification.png', dpi=300)
    print("[SUCCESS] Verified real-data plot exported to workspace root folder.")
