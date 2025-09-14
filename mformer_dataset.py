"""
# mformer dataset sampling and analysis
#
# USAGE:
# 1. To run with mock data (as before):
#    uv run mformer_dataset.py
#
# 2. To analyze a real encodec artifact:
#    uv run mformer_dataset.py --analyze_npz data/measureformer/mproj_ii.npz --input_audio data/measureformer/MORGPROJ_II.wav
#
"""
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
import torch
from tqdm import tqdm
import os
import uuid
from datetime import datetime
import pyarrow
import argparse # NEW: For command-line arguments

# NEW: Lazy imports for audio/plotting to keep stub mode dependency-light
_plt = None
_torchaudio = None

from scipy.signal import stft

# =============================================================================
# NEW: Principled Hyperparameters
# =============================================================================
class Hyperparameters:
    # --- Musical Structure ---
    MAX_TEMPO_BPM = 165.0
    BEATS_PER_MEASURE = 4
    MEASURES_PER_CHUNK = 2.0

    # --- Analysis Timescales (in seconds) ---
    STABILITY_WINDOW_SEC = 6.0

    # In class Hyperparameters:
    FAST_EMA_SEC = 4.0         # A musical phrase (~2 measures at 120bpm)
    SLOW_EMA_SEC = 90.0        # A musical section or long-form idea

  

    # --- Silence Definition ---
    SILENCE_THRESHOLD_DB = -50.0

    # --- DataLoader Settings ---
    BATCH_SIZE = 8
    SEQUENCE_LENGTH_TOKENS = 512

    # --- Encodec Specification ---
    ENCODEC_FRAME_RATE = 75.0
    # The sample rate the Encodec model was trained on and operates at.
    ENCODEC_SAMPLE_RATE = 24000

def calculate_chunk_size(params: Hyperparameters) -> int:
    """Calculates chunk size in tokens based on musical tempo."""
    beats_per_second = params.MAX_TEMPO_BPM / 60.0
    measures_per_second = beats_per_second / params.BEATS_PER_MEASURE
    seconds_per_chunk = params.MEASURES_PER_CHUNK / measures_per_second
    chunk_size_tokens = int(seconds_per_chunk * params.ENCODEC_FRAME_RATE)
    print(f"Calculated chunk size: {chunk_size_tokens} tokens (~{seconds_per_chunk:.2f} sec)")
    return chunk_size_tokens

def amplitude_to_db(amplitude, ref=1.0, epsilon=1e-9):
    """Converts linear amplitude to decibels, handling zeros safely."""
    return 20 * np.log10(np.maximum(amplitude, epsilon) / ref)

# =============================================================================
# PHASE 1 STUB: Mock Data Generation
# =============================================================================
def generate_mock_data(num_chunks=2000, feature_dim=3, chunk_size_tokens=150):
    """Generates a mock feature time series and token array to simulate a recording."""
    print("PHASE 1 (STUB): Generating mock feature data and token stream...")
    features = np.zeros((num_chunks, feature_dim), dtype=np.float32)
    
    # SILENCE: Set RMS to a very low linear amplitude
    features[0:200, 0] = 10**(Hyperparameters.SILENCE_THRESHOLD_DB / 20) * 0.1
    features[0:200, 2] = 0.1
    
    loop_pattern = np.array([0.8, 0.5, 0.3])
    features[200:1000] = loop_pattern + np.random.randn(800, feature_dim) * 0.01
    
    features[1000:1800, 0] = 0.7 + np.sin(np.arange(800) * 0.1) * 0.2
    features[1000:1800, 1] = 0.5 + np.cos(np.arange(800) * 0.05) * 0.3
    features[1000:1800, 2] = np.random.uniform(0.6, 0.9, size=(800,))

    features[1800] = np.array([0.9, 0.9, 0.95])
    
    features[1801:2000, 0] = 10**(Hyperparameters.SILENCE_THRESHOLD_DB / 20) * 0.1
    features[1801:2000, 2] = 0.1
    
    total_tokens = num_chunks * chunk_size_tokens
    tokens = np.random.randint(0, 1024, size=total_tokens, dtype=np.uint16)
    
    print(f"Generated {num_chunks} feature vectors and {total_tokens} tokens.")
    return features, tokens, Hyperparameters.ENCODEC_FRAME_RATE

# =============================================================================
# NO LONGER A STUB: Real Feature Calculation from Audio
# =============================================================================
def calculate_features_from_audio(audio_path: str, tokens_shape, frame_rate: float, chunk_size_tokens: int):
    """
    Calculates feature vectors from raw audio using full spectral analysis for
    superior detection of timbral change.
    """
    global _torchaudio
    if _torchaudio is None:
        import torchaudio
        _torchaudio = torchaudio

    print(f"Loading raw audio from {audio_path} for feature calculation...")
    wav, sr = _torchaudio.load(audio_path)

    # --- 1. Pre-process Audio ---
    if sr != Hyperparameters.ENCODEC_SAMPLE_RATE:
        resampler = _torchaudio.transforms.Resample(sr, Hyperparameters.ENCODEC_SAMPLE_RATE)
        wav = resampler(wav)
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)

    # --- 2. Naive Token-to-Audio Alignment ---
    total_samples = wav.shape[1]
    total_tokens = tokens_shape[1]
    samples_per_token = total_samples / total_tokens
    num_chunks = (total_tokens + chunk_size_tokens - 1) // chunk_size_tokens
    chunk_size_samples = int(chunk_size_tokens * samples_per_token)

    print(f"Aligning {total_samples} samples to {total_tokens} tokens.")
    
    # --- 3. Extract Per-Chunk Features (Now with Full Spectra) ---
    all_rms = []
    all_spectra = []
    wav_mono = wav.squeeze(0).numpy()

    for i in tqdm(range(num_chunks), desc="  Calculating chunk spectra"):
        start_sample = i * chunk_size_samples
        end_sample = min(total_samples, (i + 1) * chunk_size_samples)
        audio_chunk = wav_mono[start_sample:end_sample]
        
        if audio_chunk.shape[0] < 256: # Need enough samples for an STFT window
            all_rms.append(0.0)
            all_spectra.append(np.zeros(129, dtype=np.float32)) # Append zero vector of correct shape
            continue
            
        # Feature 1: RMS Energy (still fundamental)
        rms_energy = np.sqrt(np.mean(audio_chunk**2))
        all_rms.append(rms_energy)

        # High-Dimensional Feature: The Spectrum
        # Using a standard STFT configuration
        nperseg = 256 # STFT window size
        f, t, Zxx = stft(audio_chunk, fs=Hyperparameters.ENCODEC_SAMPLE_RATE, nperseg=nperseg)
        
        # Get magnitude and average across time to get one vector per chunk
        magnitude_spectrum = np.mean(np.abs(Zxx), axis=1)
        all_spectra.append(magnitude_spectrum)

    # --- 4. "Late Reduce": Calculate Flux from Spectral Vectors ---
    print("  Post-processing: Calculating vector-space spectral flux...")
    all_spectra = np.array(all_spectra)
    
    # Pre-allocate flux arrays
    shape_flux = np.zeros(num_chunks)
    mag_flux = np.zeros(num_chunks)

    # Use a tiny epsilon to avoid log(0)
    epsilon = 1e-9

    for i in range(1, num_chunks):
        # Compare chunk i to chunk i-1
        prev_spec = all_spectra[i-1]
        curr_spec = all_spectra[i]

        # Flux A: Spectral Shape Flux (Cosine Similarity)
        # 1.0 (max difference) to 0.0 (identical shape)
        shape_flux[i] = 1.0 - cosine(prev_spec + epsilon, curr_spec + epsilon)

        # Flux B: Log-Magnitude Flux (Perceptual Delta)
        # Convert to a log scale (like dB) before differencing
        prev_log_spec = np.log(prev_spec + epsilon)
        curr_log_spec = np.log(curr_spec + epsilon)
        mag_flux[i] = np.mean(np.abs(prev_log_spec - curr_log_spec))

    # --- 5. Final Feature Assembly & Normalization ---
    # Normalize each flux feature to a clean [0, 1] range
    if shape_flux.max() > 0:
        shape_flux /= shape_flux.max()
    if mag_flux.max() > 0:
        mag_flux /= mag_flux.max()
        
    # Combine into our final 3-dimensional feature vector
    features = np.stack([
        np.array(all_rms),
        shape_flux,
        mag_flux
    ], axis=1).astype(np.float32)

    print("Feature calculation complete.")
    return features


# =============================================================================
# PHASE 2: Structural Signal Generation (Refactored for Time-Based Params)
# =============================================================================
class StructuralAnalyzer:
    """Analyzes a feature time series to produce a sampling priority score."""
    def __init__(self, features, frame_rate: float, chunk_size_tokens: int, params: Hyperparameters):
        self.features = features
        self.w_novelty = 0.7
        self.w_instability = 0.3

        # --- NEW: Convert time-based params to chunk-based params ---
        chunks_per_second = frame_rate / chunk_size_tokens
        self.stability_window_size = max(2, int(params.STABILITY_WINDOW_SEC * chunks_per_second))
        self.fast_ema_span = max(1, int(params.FAST_EMA_SEC * chunks_per_second))
        self.slow_ema_span = max(1, int(params.SLOW_EMA_SEC * chunks_per_second))

        print("\n--- Structural Analyzer Initialized ---")
        print(f"  - Stability Window: {self.stability_window_size} chunks (~{params.STABILITY_WINDOW_SEC}s)")
        print(f"  - EMA Spans (Fast/Slow): {self.fast_ema_span} / {self.slow_ema_span} chunks")


    def _compute_stability(self):
        """Computes local self-similarity via a sliding window of cosine distances."""
        num_chunks = len(self.features)
        stability_signal = np.ones(num_chunks)
        for i in tqdm(range(num_chunks), desc="  Computing stability"):
            start = max(0, i - self.stability_window_size // 2)
            end = min(num_chunks, i + self.stability_window_size // 2)
            window = self.features[start:end]
            center_vec = self.features[i]
            similarities = 1 - np.array([cosine(center_vec, v) for v in window])
            stability_signal[i] = np.mean(similarities)
        return stability_signal

    def _compute_vector_novelty(self):
        """Computes novelty based on the magnitude of a vector EMA crossover."""
        df = pd.DataFrame(self.features)
        fast_ema = df.ewm(span=self.fast_ema_span, adjust=False).mean().values
        slow_ema = df.ewm(span=self.slow_ema_span, adjust=False).mean().values
        
        novelty_vectors = fast_ema - slow_ema
        scalar_novelty = np.linalg.norm(novelty_vectors, axis=1)
        
        return (scalar_novelty - scalar_novelty.min()) / (scalar_novelty.max() - scalar_novelty.min())

    def analyze(self):
        """Runs the full analysis pipeline."""
        print("PHASE 2: Generating structural signals...")
        stability = self._compute_stability()
        novelty = self._compute_vector_novelty()
        
        instability = (1 - stability)**2
        priority_scores = (self.w_novelty * novelty) + (self.w_instability * instability)
        
        priority_scores[priority_scores < 0] = 0
        if priority_scores.sum() == 0:
            priority_scores = np.ones_like(priority_scores)
            
        print("Structural analysis complete.")
        return priority_scores, stability, novelty

# =============================================================================
# PHASE 3: Intelligent Sampling and Loading (Refactored for dB Threshold)
# =============================================================================
class IntelligentAudioLoader:
    def __init__(self, tokens, features, frame_rate: float, chunk_size_tokens: int, params: Hyperparameters,
                 sampler_mode='DYNAMIC_PRIORITY', priority_scores=None):
        
        self.tokens = torch.from_numpy(tokens)
        self.features = features
        self.batch_size = params.BATCH_SIZE
        self.sequence_length = params.SEQUENCE_LENGTH_TOKENS
        self.num_chunks = len(features)
        self.chunk_size = chunk_size_tokens
        self.sampler_mode = sampler_mode

        # NEW: Convert linear amplitude to dB and apply threshold
        rms_db = amplitude_to_db(self.features[:, 0])
        self.is_void = rms_db < params.SILENCE_THRESHOLD_DB
        self.void_to_music_transitions = self._find_transitions()
        
        print("\n--- DataLoader Initialized ---")
        print(f"  - Silence Threshold: < {params.SILENCE_THRESHOLD_DB} dB")
        print(f"  - Found {np.sum(self.is_void)} silent chunks ({np.sum(self.is_void)/self.num_chunks:.1%})")
        print(f"  - Found {len(self.void_to_music_transitions)} silence-to-music transitions.")

        if sampler_mode == 'UNIFORM':
            self.weights = np.ones(self.num_chunks)
        elif sampler_mode == 'DYNAMIC_PRIORITY':
            if priority_scores is None:
                raise ValueError("priority_scores must be provided for DYNAMIC_PRIORITY mode.")
            self.weights = priority_scores
        else:
            raise ValueError(f"Unknown sampler_mode: {sampler_mode}")

        self.weights[self.weights < 0] = 0
        if self.weights.sum() > 0:
            self.weights /= self.weights.sum()
        else:
            self.weights = np.ones(self.num_chunks) / self.num_chunks

    def _find_transitions(self):
        """Finds all chunk indices where a void chunk is followed by a non-void one."""
        shifted_void = np.roll(self.is_void, -1)
        transitions = np.where(self.is_void & ~shifted_void)[0]
        return transitions + 1

    def __iter__(self):
        return self

    def __next__(self):
        """Generates one batch of data using the Propose-Validate-Remediate loop."""
        batch_x, batch_y = [], []
        num_chunks_in_sequence = (self.sequence_length + self.chunk_size - 1) // self.chunk_size

        for _ in range(self.batch_size):
            while True:
                candidate_chunk_idx = np.random.choice(self.num_chunks, p=self.weights)
                end_chunk = min(self.num_chunks, candidate_chunk_idx + num_chunks_in_sequence)
                chunk_window = np.arange(candidate_chunk_idx, end_chunk)

                if not np.all(self.is_void[chunk_window]):
                    final_chunk_idx = candidate_chunk_idx
                    break
                else:
                    next_transitions = self.void_to_music_transitions[self.void_to_music_transitions > candidate_chunk_idx]
                    if len(next_transitions) > 0:
                        remediated_chunk_idx = next_transitions[0]
                    else:
                        if len(self.void_to_music_transitions) > 0:
                            remediated_chunk_idx = self.void_to_music_transitions[0]
                        else: # No transitions at all, just pick a random non-void chunk
                            non_void_indices = np.where(~self.is_void)[0]
                            remediated_chunk_idx = np.random.choice(non_void_indices) if len(non_void_indices) > 0 else candidate_chunk_idx
                    
                    final_chunk_idx = remediated_chunk_idx
                    break
            
            start_token = (final_chunk_idx * self.chunk_size) - (self.sequence_length // 2)
            start_token = max(0, start_token)
            end_token = start_token + self.sequence_length + 1
            
            if end_token > len(self.tokens):
                end_token = len(self.tokens)
                start_token = end_token - (self.sequence_length + 1)
            
            token_slice = self.tokens[start_token:end_token].long()
            batch_x.append(token_slice[:-1])
            batch_y.append(token_slice[1:])

        return torch.stack(batch_x), torch.stack(batch_y)

# =============================================================================
# Main Execution Block: Now with Argument Parsing
# =============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run structural analysis and sampling on audio token data."
    )
    parser.add_argument(
        "--analyze_npz",
        type=str,
        default=None,
        help="Path to a real .npz artifact to analyze instead of using mock data."
    )
    parser.add_argument(
        "--input_audio",
        type=str,
        default=None,
        help="Path to the corresponding raw audio file (required if --analyze_npz is used)."
    )
    args = parser.parse_args()

    # --- Setup ---
    params = Hyperparameters()
    chunk_size_tokens = calculate_chunk_size(params)
    
    if args.analyze_npz:
        if not args.input_audio:
            raise ValueError("--input_audio is required when using --analyze_npz.")
        
        print(f"\n--- Running in ANALYSIS mode on {args.analyze_npz} ---")
        # --- PHASE 1: Load Real Data ---
        data = np.load(args.analyze_npz)
        tokens = data['tokens']
        # For simplicity, we'll use the first codebook stream for dataloading shape.
        # The BPE process would handle the multi-stream nature more formally.
        tokens_for_loader = tokens[0, :]
        frame_rate = data['frame_rate'].item()
        
        features = calculate_features_from_audio(
            args.input_audio, tokens.shape, frame_rate, chunk_size_tokens
        )
        
    else:
        print("\n--- Running in MOCK mode ---")
        # --- PHASE 1 (STUB) ---
        features, tokens_for_loader, frame_rate = generate_mock_data(chunk_size_tokens=chunk_size_tokens)
    
    # --- PHASE 2 ---
    analyzer = StructuralAnalyzer(features, frame_rate, chunk_size_tokens, params)
    priority_scores, stability, novelty = analyzer.analyze()
    
    # --- File Naming and Saving ---
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    unique_id = str(uuid.uuid4())[:4]
    filename_stem = f"{timestamp}_{unique_id}"
    
    LOG_DIR_PARQUET = 'data/measureformer_analysis'
    os.makedirs(LOG_DIR_PARQUET, exist_ok=True)
    parquet_filepath = os.path.join(LOG_DIR_PARQUET, f"{filename_stem}.parquet")

    metadata_df = pd.DataFrame({
        'rms_energy': features[:, 0],
        'shape_flux': features[:, 1], # NEW, more powerful feature
        'magnitude_flux': features[:, 2], # NEW, more powerful feature
        'stability': stability,
        'novelty': novelty,
        'priority_score': priority_scores
    })
    
    print(f"\nSaving analysis metadata to {parquet_filepath}...")
    metadata_df.to_parquet(parquet_filepath)
    
    # --- PHASE 3 ---
    print("\nPHASE 3: Initializing DataLoader with 'DYNAMIC_PRIORITY' sampler...")
    loader = IntelligentAudioLoader(
        tokens=tokens_for_loader,
        features=features,
        frame_rate=frame_rate,
        chunk_size_tokens=chunk_size_tokens,
        params=params,
        sampler_mode='DYNAMIC_PRIORITY',
        priority_scores=priority_scores
    )
    
    print("Fetching one batch of data...")
    x, y = next(loader)
    
    print("\n--- PIPELINE COMPLETE ---")
    print(f"Successfully generated a batch of data.")
    print(f"Input tensor shape (x): {x.shape}")
    print(f"Target tensor shape (y): {y.shape}")

    # --- Visualization ---
    try:
        import matplotlib.pyplot as plt
        _plt = plt
        
        fig, axs = plt.subplots(4, 1, figsize=(15, 12), sharex=True) # Increased height for clarity
        
        # --- SUBPLOT 1: NEW INPUT FEATURES ---
        ax0_twin = axs[0].twinx()
        
        # Plot RMS on the left axis (no change)
        p1, = axs[0].plot(amplitude_to_db(features[:, 0]), label='RMS Energy (dB)', color='cyan', alpha=0.8)
        
        # Plot our TWO new flux features on the right axis
        p2, = ax0_twin.plot(features[:, 1], label='Shape Flux (Cosine Sim)', color='magenta', alpha=0.9, linewidth=1.5)
        p3, = ax0_twin.plot(features[:, 2], label='Magnitude Flux (Log Delta)', color='orange', alpha=0.7)

        axs[0].set_title('Input Features (Post-Calculation)')
        axs[0].set_ylabel('dBFS')
        ax0_twin.set_ylabel('Normalized Flux Value')

        # A robust way to combine legends from both axes
        axs[0].legend(handles=[p1, p2, p3], loc='upper left')
        
        # --- SUBPLOT 2: STABILITY ---
        axs[1].plot(stability, label='Stability Signal', color='green')
        axs[1].set_title('Phase 2: Stability Signal (Higher is more repetitive)')
        axs[1].legend()

        # --- SUBPLOT 3: NOVELTY ---
        axs[2].plot(novelty, label='Vector Novelty Signal', color='purple')
        axs[2].set_title('Phase 2: Vector Novelty Signal (Spikes indicate change)')
        axs[2].legend()
        
        # --- SUBPLOT 4: FINAL PRIORITY ---
        axs[3].plot(priority_scores, label='Final Priority Score', color='red')
        axs[3].set_title('Phase 2: Final Sampling Priority Score (Higher is more valuable)')
        axs[3].legend()
        
        plt.tight_layout()
        
        LOG_DIR_PLOTS = 'logs'
        os.makedirs(LOG_DIR_PLOTS, exist_ok=True)
        plot_filepath = os.path.join(LOG_DIR_PLOTS, f"{filename_stem}.png")

        print(f"\nSaving analysis plot to {plot_filepath}...")
        plt.savefig(plot_filepath)
        plt.close(fig)
        print("Plot saved.")

    except ImportError:
        print("\nMatplotlib not found. Skipping visualization.")