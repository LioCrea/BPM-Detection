import os
import time
import warnings
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Reduce verbose decoder warnings (mpg123, etc.)
warnings.filterwarnings("ignore", category=UserWarning)

# Use ONE hop length everywhere for consistency
HOP = 256  

def snap_into_range(bpm, bpm_min, bpm_max):
    """
    Fold BPM by powers of 2 to fall inside [bpm_min, bpm_max].
    Pick the candidate closest to the window center.
    If none enters the window, clamp to the nearest boundary.
    """
    if bpm <= 0 or bpm_min <= 0 or bpm_max <= 0 or bpm_min >= bpm_max:
        return float(bpm)

    mid = 0.5 * (bpm_min + bpm_max)
    candidates = [bpm * (2.0 ** k) for k in range(-3, 4)]  # bpm/8 

    in_range = [c for c in candidates if bpm_min <= c <= bpm_max]
    if in_range:
        return float(min(in_range, key=lambda c: abs(c - mid)))

    closest = min(candidates, key=lambda c: min(abs(c - bpm_min), abs(c - bpm_max)))
    return float(max(bpm_min, min(bpm_max, closest)))

def detect_bpm_fast(file_path,
                    duration=75.0,         
                    hop_length=HOP,        
                    start_bpm=120.0,       # initial guess
                    bpm_min=80.0, bpm_max=110.0,
                    use_hpss=True):
    """
    Fast & reliable BPM estimation:
    - partial loading (duration seconds)
    - optional HPSS to stabilize onsets
    - octave correction (×2 / ÷2) into [bpm_min, bpm_max]
    Returns: (bpm_corrected, beat_times, sr, y, onset_env)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Partial loading for speed; set sr=None to keep native sample rate
    y, sr = librosa.load(file_path, sr=None, duration=duration)

    # Optional percussive isolation to stabilize onset envelope
    if use_hpss:
        _, y_perc = librosa.effects.hpss(y)
    else:
        y_perc = y


    onset_env = librosa.onset.onset_strength(y=y_perc, sr=sr, hop_length=hop_length, aggregate=np.median)

    # Initial tempo and beats
    tempo_bt, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env,
                                                    sr=sr,
                                                    hop_length=hop_length,
                                                    start_bpm=start_bpm,
                                                    tightness=100)
    if isinstance(tempo_bt, np.ndarray):
        tempo_bt = float(tempo_bt[0])
    tempo_bt = float(tempo_bt)

    tempo_folded = snap_into_range(tempo_bt, bpm_min, bpm_max)

    tempo_final, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env,
                                                       sr=sr,
                                                       hop_length=hop_length,
                                                       start_bpm=tempo_folded,
                                                       tightness=120)
    if isinstance(tempo_final, np.ndarray):
        tempo_final = float(tempo_final[0])
    tempo_final = float(tempo_final)
    tempo_final = snap_into_range(tempo_final, bpm_min, bpm_max)

    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
    return tempo_final, beat_times, sr, y, onset_env

def plot_quick(y, sr, beat_times, onset_env, hop_length, bpm, show_plots=True, save_dir=None):

    figs = []

    # Waveform + beats
    fig1 = plt.figure(figsize=(12, 3))
    librosa.display.waveshow(y, sr=sr)
    for t in beat_times:
        plt.axvline(x=t, alpha=0.6)
    plt.title(f"Waveform + beats | BPM ≈ {bpm:.1f}")
    plt.xlabel("Time (s)")
    plt.tight_layout()
    figs.append(fig1)

    # Onset envelope + beats
    fig2 = plt.figure(figsize=(12, 3))
    times_env = librosa.times_like(onset_env, sr=sr, hop_length=hop_length)
    plt.plot(times_env, onset_env)
    ymin, ymax = float(np.min(onset_env)), float(np.max(onset_env))
    plt.vlines(beat_times, ymin=ymin, ymax=ymax, alpha=0.6)
    plt.title("Onset envelope + beats")
    plt.xlabel("Time (s)")
    plt.ylabel("Onset strength")
    plt.tight_layout()
    figs.append(fig2)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        for i, f in enumerate(figs, start=1):
            f.savefig(os.path.join(save_dir, f"plot_{stamp}_{i}.png"), dpi=150, bbox_inches="tight")
        plt.close("all")
    elif show_plots:
        plt.show()
    else:
        plt.close("all")

def plot_waveform_spotify(y, sr, show_plots=True, save_path=None):

    spotify_green = "#1DB954"
    black = "#000000"

    fig = plt.figure(figsize=(12, 3), facecolor=black)
    ax = fig.add_subplot(111)
    ax.set_facecolor(black)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(spotify_green)
    ax.spines["left"].set_color(spotify_green)
    ax.tick_params(axis="x", colors=spotify_green, labelsize=9)
    ax.tick_params(axis="y", colors=spotify_green, labelsize=9)

    # Waveform
    librosa.display.waveshow(y, sr=sr, color=spotify_green, ax=ax)
    ax.set_xlabel("Time (s)", color=spotify_green)
    ax.set_ylabel("Amplitude", color=spotify_green)
    plt.margins(x=0)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
    elif show_plots:
        plt.show()
    else:
        plt.close(fig)

def detect_fundamental_pitch(y, sr, use_hpss=True, fmin_note="C1", fmax_note="C7",
                             frame_length=2048, hop_length=HOP):

    y_in = librosa.effects.hpss(y)[0] if use_hpss else y

    fmin = librosa.note_to_hz(fmin_note)
    fmax = librosa.note_to_hz(fmax_note)

    f0 = librosa.yin(y_in, fmin=fmin, fmax=fmax, sr=sr,
                     frame_length=frame_length, hop_length=hop_length)

    f0 = f0[np.isfinite(f0) & (f0 > 0)]
    if len(f0) == 0:
        return None, None

    fundamental_freq = float(np.median(f0))
    note_name = librosa.hz_to_note(fundamental_freq) 
    return fundamental_freq, note_name


def estimate_key_ks(y, sr, use_hpss=True):

    y_in = librosa.effects.hpss(y)[0] if use_hpss else y

    chroma = librosa.feature.chroma_cqt(y=y_in, sr=sr)
    chroma_vec = chroma.mean(axis=1)  

    if np.allclose(chroma_vec.sum(), 0):
        return None, None, 0.0
    chroma_vec = chroma_vec / (np.linalg.norm(chroma_vec) + 1e-12)

    prof_major = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                           2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=float)
    prof_minor = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                           2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=float)

    prof_major /= np.linalg.norm(prof_major)
    prof_minor /= np.linalg.norm(prof_minor)

    pitch_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    best = {"mode": None, "key_idx": None, "score": -np.inf}

    for k in range(12):
        maj_score = float(np.dot(chroma_vec, np.roll(prof_major, k)))
        min_score = float(np.dot(chroma_vec, np.roll(prof_minor, k)))
        if maj_score > best["score"]:
            best.update({"mode": "major", "key_idx": k, "score": maj_score})
        if min_score > best["score"]:
            best.update({"mode": "minor", "key_idx": k, "score": min_score})

    maj_score_k = float(np.dot(chroma_vec, np.roll(prof_major, best["key_idx"])))
    min_score_k = float(np.dot(chroma_vec, np.roll(prof_minor, best["key_idx"])))
    other = min_score_k if best["mode"] == "major" else maj_score_k
    confidence = max(0.0, best["score"] - other)  

    key_name = pitch_names[best["key_idx"]]
    return key_name, best["mode"], confidence




if __name__ == "__main__":
    # Specify path here (mp3 and should work with wav too)
    file_path = "Ride On - C.S. Armstrong.mp3"

    # For better use, create a window between the BPM (ex is here given for a 150ish track)
    BPM_MIN = 80.0
    BPM_MAX = 120.0

    bpm, beats, sr, y, onset_env = detect_bpm_fast(
        file_path,
        duration=300,       # TODO get track length
        hop_length=HOP,
        start_bpm=BPM_MIN,      
        bpm_min=BPM_MIN, bpm_max=BPM_MAX,
        use_hpss=True
    )

    print(f"- Estimated BPM (corrected): {bpm:.2f}")
    print(f"- Detected beats: {len(beats)}")

    plot_quick(y, sr, beats, onset_env, hop_length=HOP, bpm=bpm,
               show_plots=True,     
               save_dir=None)       

    save_path = None 
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plot_waveform_spotify(y, sr, show_plots=True, save_path=save_path)


    # --- NEW: note perfect yet ---
    fund_freq, fund_note = detect_fundamental_pitch(
        y, sr, use_hpss=True, fmin_note="C1", fmax_note="C7",
        frame_length=2048, hop_length=HOP
    )
    if fund_note:
        print(f"Fundamental pitch: {fund_freq:.2f} Hz ({fund_note})")
    else:
        print("Fundamental pitch: not stable / not found")

    key_name, mode, conf = estimate_key_ks(y, sr, use_hpss=True)
    if key_name:
        print(f"Estimated key: {key_name} {mode} (confidence margin: {conf:.3f})")
    else:
        print("Estimated key: not found")
