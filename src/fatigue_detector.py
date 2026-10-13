import os
import numpy as np
import matplotlib.pyplot as plt
from signal_processing import process_emg_pipeline
from data_loader import fetch_realworld_emg_data

class NeuroSafetyInterlock:
    def __init__(self, baseline_mdf_windows=4, threshold_drop_pct=0.15, trend_patience=3):
        """
        Advanced Cybernetic Safety Controller.
        Triggers if absolute threshold is breached OR if a severe downward trend is sustained.
        """
        self.baseline_windows = baseline_mdf_windows
        self.threshold_pct = threshold_drop_pct
        self.patience = trend_patience
        
    def monitor_stream(self, time_vectors, mdf_trajectory):
        """Scans trajectories using multi-window predictive tracking."""
        baseline_mdf = np.mean(mdf_trajectory[:self.baseline_windows])
        critical_threshold = baseline_mdf * (1 - self.threshold_pct)
        
        interlock_fired = False
        interlock_time = None
        trigger_index = -1
        
        # Continuous monitoring sweep
        for idx in range(self.baseline_windows, len(mdf_trajectory)):
            current_val = mdf_trajectory[idx]
            
            # Condition 1: Direct threshold breach
            if current_val <= critical_threshold:
                interlock_fired = True
                interlock_time = time_vectors[idx]
                trigger_index = idx
                break
            
            # Condition 2: Sustained localized decay trend (Predictive Alert)
            if idx >= self.baseline_windows + self.patience:
                recent_windows = mdf_trajectory[idx - self.patience : idx + 1]
                # If the frequency drops consistently across consecutive steps
                if np.all(np.diff(recent_windows) < 0) and current_val < (baseline_mdf * 0.95):
                    interlock_fired = True
                    interlock_time = time_vectors[idx]
                    trigger_index = idx
                    break
                
        return baseline_mdf, critical_threshold, interlock_fired, interlock_time, trigger_index

if __name__ == "__main__":
    fs = 1000
    
    print("[PIPELINE] Initializing data acquisition loop...")
    time, raw_emg = fetch_realworld_emg_data()
    
    print("[PIPELINE] Running sliding-window spectral features...")
    feature_time, mdf_track, rms_track, clean_emg = process_emg_pipeline(raw_emg, fs)
    
    # Evaluate using the advanced trend model
    detector = NeuroSafetyInterlock(baseline_mdf_windows=4, threshold_drop_pct=0.15, trend_patience=3)
    base_f, thresh_f, fired, f_time, f_idx = detector.monitor_stream(feature_time, mdf_track)
    
    print(f"\n--- Advanced Cybernetic State Estimator Results ---")
    print(f"Baseline Median Frequency: {base_f:.2f} Hz")
    print(f"Safety Cutoff Threshold: {thresh_f:.2f} Hz (15% Drop)")
    
    if fired:
        print(f"STATUS: INTERLOCK ENGAGED at {f_time:.2f} seconds!")
        print(f"Reason: Predictive trend analysis detected progressive neuromuscular drift.")
    else:
        print("STATUS: System Nominal. Signal variance within safe operational margins.")
    print("-----------------------------------------------------")
    
    # Render the polished repository diagnostic chart
    plt.figure(figsize=(11, 5.5))
    plt.plot(feature_time, mdf_track, marker='o', color='#1a1a1a', linewidth=1.5, label='Tracked Median Frequency (MDF)')
    plt.axhline(y=base_f, color='#2e7d32', linestyle='--', linewidth=1.5, label=f'Nominal Baseline ({base_f:.1f} Hz)')
    plt.axhline(y=thresh_f, color='#d32f2f', linestyle='--', linewidth=1.5, label=f'Critical Limit ({thresh_f:.1f} Hz)')
    
    if fired:
        plt.axvline(x=f_time, color='#ffcdd2', alpha=0.5, linestyle='-', zorder=1)
        plt.scatter(f_time, mdf_track[f_idx], color='#d32f2f', s=180, zorder=5, 
                    label=f'Safety Interlock Fired ({f_time:.2f}s)')
        
    plt.title("NeuroSensorimotor-Predict: Real-Time Fatigue State Estimation", fontsize=12, fontweight='bold')
    plt.xlabel("Execution Time (seconds)", fontsize=10)
    plt.ylabel("Spectral Center of Mass (Hz)", fontsize=10)
    plt.legend(loc='lower left', frameon=True)
    plt.grid(True, alpha=0.3, linestyle=':')
    plt.tight_layout()
    
    plt.savefig('predictive_interlock_verification.png', dpi=300)
    print("[SUCCESS] Polished plot updated in the workspace root directory.")
