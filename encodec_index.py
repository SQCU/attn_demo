# encodec_index.py
# uv run encodec_index.py --input_audio data\measureformer\MORGPROJ_II.wav --output_npz data\measureformer\mproj_ii.npz
import torch
import torchaudio
import numpy as np
import argparse
import os
from tqdm import tqdm

# UPDATED: Correct import based on the official documentation
from encodec import EncodecModel
from encodec.utils import convert_audio

def find_optimal_split_points(
    wav: torch.Tensor,
    sample_rate: int,
    target_num_chunks: int = 16,
    min_silence_sec: float = 1.0,
    silence_thresh_db: float = -60.0,
) -> list[int]:
    """
    Finds intelligent split points using an iterative refinement strategy.
    It repeatedly finds the largest current chunk and attempts to split it
    in the middle of a long silent section. If no suitable silent point is
    found within that chunk, it splits that chunk uniformly in half.
    This process is repeated until the target number of chunks is reached.
    """
    print("Finding optimal split points using iterative refinement...")
    total_samples = wav.shape[-1]
    min_silence_samples = int(min_silence_sec * sample_rate)

    # Pre-compute all potential silent points for efficiency
    window_size = 2048
    step_size = 512
    num_windows = (total_samples - window_size) // step_size
    is_silent_window = np.zeros(num_windows, dtype=bool)
    
    # Check windows for silence
    for i in tqdm(range(num_windows), desc="  Detecting silence"):
        window = wav[:, i * step_size : i * step_size + window_size]
        amplitude = torch.abs(window).max()
        if 20 * torch.log10(amplitude + 1e-8) < silence_thresh_db:
            is_silent_window[i] = True

    candidate_split_points = []
    in_silence = False
    silence_start_idx = 0
    for i, is_silent in enumerate(is_silent_window):
        if is_silent and not in_silence:
            in_silence = True
            silence_start_idx = i
        elif not is_silent and in_silence:
            in_silence = False
            silence_duration_windows = i - silence_start_idx
            if silence_duration_windows * step_size > min_silence_samples:
                mid_point_window = silence_start_idx + silence_duration_windows // 2
                mid_point_sample = mid_point_window * step_size + window_size // 2
                candidate_split_points.append(mid_point_sample)
    
    candidate_split_points = np.array(candidate_split_points)
    print(f"  Found {len(candidate_split_points)} candidate silent split points.")

    # Iteratively split the largest chunk until we reach the target number
    splits = [0, total_samples]
    
    pbar = tqdm(total=target_num_chunks, desc="  Refining splits")
    pbar.update(1) # We start with 1 chunk

    while len(splits) < target_num_chunks + 1:
        chunk_sizes = np.diff(splits)
        largest_chunk_idx = np.argmax(chunk_sizes)
        
        start = splits[largest_chunk_idx]
        end = splits[largest_chunk_idx+1]
        midpoint = start + (end - start) // 2
        
        # Find candidate silent points within the largest chunk
        local_candidates = candidate_split_points[
            (candidate_split_points > start) & (candidate_split_points < end)
        ]

        if len(local_candidates) > 0:
            # Smart split: find the silent point closest to the middle
            best_split_point = local_candidates[np.argmin(np.abs(local_candidates - midpoint))]
            splits.append(best_split_point)
        else:
            # Fallback: split this specific chunk uniformly
            splits.append(midpoint)
        
        splits.sort()
        pbar.update(1)
    
    pbar.close()
    print(f"  Successfully created {len(splits) - 1} chunks.")
    return splits


def create_encodec_index(input_audio_path: str, output_npz_path: str):
    """
    Implements Phase 1 with intelligent, silence-based chunking for large audio files.
    """
    print("--- Starting Phase 1: Canonical Multi-Stream Tokenization ---")

    # (Model and audio loading logic is unchanged...)
    print("Loading Encodec 24kHz model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = EncodecModel.encodec_model_24khz()
    target_bandwidth = 6.0
    model.set_target_bandwidth(target_bandwidth)
    model.to(device)
    model.eval()

    print(f"Loading audio from: {input_audio_path}")
    wav, sr = torchaudio.load(input_audio_path)
    wav = convert_audio(wav, sr, model.sample_rate, model.channels)
    
    duration_sec = wav.shape[-1] / model.sample_rate
    print(f"Audio duration: {duration_sec:.2f} seconds.")
    
    # NEW: Implement the more aggressive 1/16^2 slicing strategy
    target_chunks = 256 # 16 * 16
    print(f"Using aggressive slicing with target of {target_chunks} chunks.")
    split_points = find_optimal_split_points(wav, model.sample_rate, target_num_chunks=target_chunks)

    cudnn_original_state = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False

    print(f"Processing audio in {len(split_points) - 1} chunks...")
    all_codes = []

    progress_bar = tqdm(range(len(split_points) - 1), unit='chunk', desc="Encoding Chunks")
    try:
        with torch.no_grad():
            for i in progress_bar:
                start_sample = split_points[i]
                end_sample = split_points[i+1]
                wav_chunk = wav[:, start_sample:end_sample]
                
                if wav_chunk.shape[-1] == 0:
                    continue

                wav_chunk = wav_chunk.contiguous().unsqueeze(0).to(device)
                
                encoded_frames = model.encode(wav_chunk)
                codes_chunk = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)
                all_codes.append(codes_chunk.cpu())

                # NEW: Explicitly clear the CUDA cache after processing each chunk
                # This prevents memory from accumulating between iterations.
                if device == 'cuda':
                    torch.cuda.empty_cache()

    finally:
        torch.backends.cudnn.enabled = cudnn_original_state
        print("\nRestored original cuDNN state.")

    # (The rest of the function for concatenation and saving remains the same...)
    codes = torch.cat(all_codes, dim=-1)
    multi_stream_tokens = codes.squeeze(0).numpy()

    num_codebooks, num_frames = multi_stream_tokens.shape
    print(f"Encoding complete. Token matrix shape: {multi_stream_tokens.shape}")
    
    metadata = {
    'sample_rate': np.array(model.sample_rate),
    'frame_rate': np.array(model.frame_rate),
    'target_bandwidth': np.array(target_bandwidth),
    }
    print(f"Metadata collected: sample_rate={metadata['sample_rate']}, frame_rate={metadata['frame_rate']}")

    output_dir = os.path.dirname(output_npz_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving canonical data to: {output_npz_path}")
    np.savez_compressed(
        output_npz_path,
        tokens=multi_stream_tokens,
        **metadata
    )
    print("Canonical data artifact saved successfully.")

    
def validate_index(npz_path: str):
    """
    Performs a loose validation on the generated .npz artifact with enhanced logging.
    """
    print(f"\n--- Validating Index File: {npz_path} ---")
    try:
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Validation failed: File does not exist at {npz_path}")

        data = np.load(npz_path)

        expected_keys = ['tokens', 'sample_rate', 'frame_rate', 'target_bandwidth']
        for key in expected_keys:
            if key not in data:
                raise KeyError(f"Validation failed: Missing required key '{key}' in the archive.")
        print("✅ All required keys are present.")

        tokens = data['tokens']
        # (Token validation is unchanged)
        if not isinstance(tokens, np.ndarray) or tokens.ndim != 2 or not np.issubdtype(tokens.dtype, np.integer):
            raise TypeError("Validation failed: 'tokens' is not a valid 2D integer NumPy array.")
        print(f"✅ 'tokens' array is a valid 2D integer NumPy array with shape {tokens.shape}.")

        # --- ENHANCED LOGGING AND FIX ---
        for key in ['sample_rate', 'frame_rate', 'target_bandwidth']:
            value = data[key]
            # This new print statement will reveal the type issue.
            print(f"  Inspecting '{key}': value={value}, type={type(value)}")
            
            # The fix is to check if it's either a scalar OR a 0-D array.
            # The item() method on a 0-D array extracts the scalar.
            if not (np.isscalar(value) or (isinstance(value, np.ndarray) and value.ndim == 0)):
                 raise TypeError(f"Validation failed: Metadata key '{key}' should be a scalar value.")
        print("✅ Metadata values are valid scalars or 0-D arrays.")

        print("\nValidation successful. The index artifact is well-formed.")

    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Phase 1: Create a canonical multi-stream token index from a raw audio file using Encodec."
    )
    parser.add_argument(
        "--input_audio",
        type=str,
        required=True,
        help="Path to the source audio file (.wav, .flac, etc.)."
    )
    parser.add_argument(
        "--output_npz",
        type=str,
        required=True,
        help="Path to save the output canonical data artifact (.npz file)."
    )

    args = parser.parse_args()

    create_encodec_index(args.input_audio, args.output_npz)
    validate_index(args.output_npz)