# BPM-Detection
A fast way to estimate any music BPM (+ visualization)

Fast BPM Detection & Spotify-Style Waveform Visualization

A lightweight Python tool to analyze a track’s BPM, detect beat positions, and generate clean audio visualizations — including a green-on-black waveform inspired by Spotify’s design.

Features

Fast BPM detection
Analyzes only a short portion of the track (default 60–90 seconds) for quick and reliable tempo estimation.

Automatic tempo correction
Fixes octave errors (e.g. detecting 95 → 190 → 95 BPM).

HPSS (Harmonic–Percussive Source Separation)
Optionally isolates the percussive content to improve beat detection stability.

Waveform and onset envelope visualization
Displays the waveform with detected beat markers, plus the onset strength curve.

Spotify-style waveform export
Clean black background with green line, exportable as a high-quality PNG.

Flexible output
Choose between interactive display (plt.show()) or silent PNG export for batch processing.

Requirements

Python ≥ 3.8

Install dependencies:

pip install librosa matplotlib numpy soundfile

File Structure
bpm_detection.py
README.md
plots/
└── waveform_spotify.png   # Example output

Usage

Place your audio file (e.g. Saw of Olympus.mp3) in the same folder as the script.

Edit the path in the script:

file_path = "Saw of Olympus.mp3"


Run the program:

python bpm_detection.py


Close all figures when done — the script will then terminate normally.

Main Parameters
Parameter	Description	Default
duration	Duration (seconds) analyzed from start of track	75.0
hop_length	Analysis frame step (smaller = more precise)	256
start_bpm	Initial guess for the beat tracker	120.0
bpm_min / bpm_max	Target tempo window for octave correction	80–110
use_hpss	Apply percussive/harmonic separation before analysis	True
Visual Outputs
1. Waveform with Beat Markers

Displays the waveform with vertical lines marking detected beats.

2. Onset Envelope with Beat Markers

Shows onset strength (attack envelope) with beat alignment.

3. Spotify-Style Waveform

A clean, minimal waveform in green (#1DB954) on black, optionally exported as a PNG.

Example:

save_path = "plots/waveform_spotify.png"
plot_waveform_spotify(y, sr, show_plots=False, save_path=save_path)

Troubleshooting

Script doesn’t finish
Close all Matplotlib windows or use show_plots=False to disable interactive mode.

Incorrect BPM values (half/double)
Adjust bpm_min / bpm_max to a range close to your track (e.g. 80–110 for 95 BPM, 140–190 for hardstyle).

Slow analysis
Reduce duration to 60 seconds and/or set hop_length=512.

Developer Notes

Modular design:

detect_bpm_fast() handles tempo analysis

plot_quick() displays waveform and onset envelope

plot_waveform_spotify() creates the green-on-black waveform

Easy to extend for:

Folder scanning and batch processing

Automatic PNG exports

Integration into Streamlit or Flask web apps
