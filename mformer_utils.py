# mformer_utils.py
import torch
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm
import ffmpeg  # <-- ADD THIS
import sys     # <-- ADD THIS

# --- Core components from your project, now in a utility module ---
from encodec import EncodecModel
from encodec.utils import convert_audio
from mformer_dataset import Hyperparameters, StructuralAnalyzer, calculate_chunk_size, calculate_features_from_audio, load_audio_with_ffmpeg

def tokenize_audio_on_the_fly(audio_path: str, model: EncodecModel) -> (np.ndarray, dict):
    """
    Takes an audio file path (any format), tokenizes it with Encodec,
    and returns the tokens and metadata in memory.
    """
    print(f"Tokenizing on-the-fly using FFmpeg: {audio_path}")
    
    # --- THIS IS THE KEY CHANGE ---
    # We completely replace torchaudio.load with our new, robust function.
    wav, sr = load_audio_with_ffmpeg(audio_path, model.sample_rate)
    # ----------------------------

    # We still use convert_audio as it correctly normalizes the waveform volume
    wav = convert_audio(wav, sr, model.sample_rate, model.channels)
    device = next(model.parameters()).device
    wav = wav.unsqueeze(0).to(device)

    cudnn_original_state = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False
    try:
        with torch.no_grad():
            encoded_frames = model.encode(wav)
        codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)
    finally:
        torch.backends.cudnn.enabled = cudnn_original_state

    tokens = codes.squeeze(0).cpu().numpy()
    metadata = {
        'sample_rate': model.sample_rate,
        'frame_rate': model.frame_rate,
    }
    print(f"Tokenization complete. Shape: {tokens.shape}")
    return tokens, metadata

def analyze_audio_on_the_fly(
    audio_path: str,
    tokens: np.ndarray,
    frame_rate: float,
    params: Hyperparameters,
    novelty_mode: str = "crossover" # You can expose other analysis params here
) -> np.ndarray:
    """
    Takes audio/token data, runs the full structural analysis, and returns
    the final priority scores.
    
    Args:
        audio_path (str): Path to the raw audio (for feature calculation).
        tokens (np.ndarray): The token array from tokenization.
        frame_rate (float): The frame rate from Encodec metadata.
        params (Hyperparameters): The hyperparameter object.
        novelty_mode (str): The novelty mode for the analyzer.

    Returns:
        np.ndarray: The final priority scores for each chunk.
    """
    print("Analyzing on-the-fly...")
    chunk_size_tokens = calculate_chunk_size(params)
    
    # 1. Calculate features from the raw audio, aligned to the token structure
    features = calculate_features_from_audio(
        audio_path, tokens.shape, frame_rate, chunk_size_tokens
    )
    
    # 2. Run the structural analyzer on the features
    analyzer = StructuralAnalyzer(features, frame_rate, chunk_size_tokens, params, novelty_mode=novelty_mode)
    priority_scores, _, _ = analyzer.analyze()
    
    print("Analysis complete.")
    return priority_scores, chunk_size_tokens