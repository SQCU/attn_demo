# prompt_utils.py
# self-test with:
# uv run prompt_utils.py
import torch
import random
import os
import numpy as np  # Added for AudioPromptGenerator
import pandas as pd # uv pip install pandas pyarrow


class PromptGenerator:
    """
    Reads a source text corpus (Parquet or plain text) and provides
    random document prefixes to seed model generation.
    """
    def __init__(self, source_file_path: str):
        print(f"Initializing PromptGenerator from {source_file_path}...")
        self.documents = self._load_documents(source_file_path)
        print(f"Loaded {len(self.documents)} documents.")

    def _load_documents(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Source file not found at: {file_path}")
        if file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
            # Assuming the text is in a column named 'text'
            if 'text' not in df.columns:
                raise ValueError("Parquet file must contain a 'text' column.")
            return df['text'].tolist()
        elif file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            # A simple heuristic for document splitting
            return [doc.strip() for doc in content.split('<|endoftext|>') if doc.strip()]
        else:
            raise ValueError("Unsupported source file format. Use .parquet or .txt")

    def get_prompts(self, num_prompts: int, prompt_length: int = 32) -> list[str]:
        """Returns a list of random prompt strings."""
        if not self.documents:
            print("Warning: No documents loaded, returning empty list.")
            return []
        if num_prompts > len(self.documents):
            print(f"Warning: Requesting more prompts ({num_prompts}) than available documents ({len(self.documents)}). Returning all available.")
            selected_docs = self.documents
        else:
            selected_docs = random.sample(self.documents, num_prompts)
        
        # Truncate each document to the desired prompt length
        prompts = [doc[:prompt_length] for doc in selected_docs]
        return prompts

class AudioPromptGenerator:
    """
    Reads a tokenized audio artifact (.npz) and its corresponding priority
    scores (.parquet) to provide random prompt sequences for model generation.
    
    This class uses the same priority-based sampling logic as the training
    data loader to ensure that evaluation prompts are representative of the
    data the model was trained on.
    """
    def __init__(self, npz_path: str, parquet_path: str):
        """
        Initializes the prompt generator by loading priority scores and
        memory-mapping the large token array.

        Args:
            npz_path (str): Path to the .npz file containing the token data.
            parquet_path (str): Path to the .parquet file with priority scores.
        """
        print(f"Initializing AudioPromptGenerator...")
        
        # 1. Load the small priority scores into memory
        print(f"  - Loading priority scores from: {parquet_path}")
        priority_df = pd.read_parquet(parquet_path)
        self.weights = torch.from_numpy(priority_df['priority_score'].values).float()
        
        if self.weights.sum() > 0:
            self.weights /= self.weights.sum()
        else:
            # Fallback to uniform distribution if all scores are zero
            self.weights = torch.ones_like(self.weights) / len(self.weights)

        # 2. Open the large token array using memory-mapping for scalability
        print(f"  - Memory-mapping tokens from: {npz_path}")
        # Assuming the tokens are in a key 'tokens' and we use the first stream
        self.tokens_memmap = np.load(npz_path, mmap_mode='r')['tokens'][0, :]

        # 3. Store essential metadata
        self.total_tokens = len(self.tokens_memmap)
        self.num_chunks = len(self.weights)
        self.chunk_size_tokens = self.total_tokens // self.num_chunks

        print(f"  - Setup complete. Found {self.total_tokens} tokens across {self.num_chunks} chunks.")

    def get_prompts(self, num_prompts: int, prompt_length: int) -> torch.Tensor:
        """
        Samples and returns a batch of prompt sequences based on audio priority.

        Args:
            num_prompts (int): The number of prompt sequences to generate.
            prompt_length (int): The length of each prompt sequence in tokens.

        Returns:
            torch.Tensor: A tensor of shape [num_prompts, prompt_length]
                          containing the token IDs for the prompts.
        """
        if prompt_length >= self.total_tokens:
            raise ValueError(f"prompt_length ({prompt_length}) cannot be larger than total_tokens ({self.total_tokens}).")

        # 1. Select high-priority chunk indices using the exact same logic as the trainer
        selected_chunk_indices = torch.multinomial(
            self.weights, 
            num_samples=num_prompts, 
            replacement=True
        )
        
        # 2. Prepare a NumPy array on the CPU to efficiently gather slices
        prompts_np = np.zeros((num_prompts, prompt_length), dtype=np.int64)

        # 3. For each selected chunk, extract a prompt sequence
        for i, chunk_idx in enumerate(selected_chunk_indices):
            # Center the prompt on the start of the interesting chunk
            start_token = (chunk_idx.item() * self.chunk_size_tokens) - (prompt_length // 2)
            
            # --- Boundary checks to prevent reading out of bounds ---
            start_token = max(0, start_token)
            end_token = start_token + prompt_length
            
            if end_token > self.total_tokens:
                end_token = self.total_tokens
                start_token = end_token - prompt_length
            
            # Read the slice directly from the memory-mapped file
            token_slice = self.tokens_memmap[start_token:end_token]
            prompts_np[i] = token_slice

        # 4. Convert the final numpy batch to a torch tensor
        return torch.from_numpy(prompts_np)

# ==============================================================================
# == Self-Testing and Demonstration Block
# ==============================================================================
if __name__ == '__main__':
    print("="*80)
    print("== Running Self-Tests for prompt_utils.py ==")
    print("="*80 + "\n")

    # --- 1. Test PromptGenerator (for text data) ---
    print("--- 1. Testing PromptGenerator (Text) ---")
    DUMMY_TXT_FILE = "prompt_utils_test.txt"
    print(f"Creating a temporary test file: {DUMMY_TXT_FILE}")
    try:
        with open(DUMMY_TXT_FILE, "w", encoding="utf-8") as f:
            f.write("This is the first document for testing purposes.<|endoftext|>")
            f.write("This is the second document, which is also used for testing.<|endoftext|>")
            f.write("A third document appears.")

        # Instantiate and test
        text_generator = PromptGenerator(source_file_path=DUMMY_TXT_FILE)
        prompts = text_generator.get_prompts(num_prompts=2, prompt_length=20)
        
        print("\n[SUCCESS] Generated text prompts:")
        for i, p in enumerate(prompts):
            print(f"  Prompt {i+1}: '{p}'")
        
        assert len(prompts) == 2
        assert all(isinstance(p, str) for p in prompts)
        assert all(len(p) <= 20 for p in prompts)
        print("[VERIFIED] Output format is correct.")

    finally:
        if os.path.exists(DUMMY_TXT_FILE):
            os.remove(DUMMY_TXT_FILE)
            print(f"\nCleaned up temporary file: {DUMMY_TXT_FILE}")

    print("\n" + "="*80 + "\n")

    # --- 2. Test AudioPromptGenerator (for tokenized audio data) ---
    print("--- 2. Testing AudioPromptGenerator (Audio Tokens) ---")
    
    # Use the same paths as your training config for consistency
    NPZ_FILE = "data/measureformer/mproj_ii.npz"
    PARQUET_FILE = "data/measureformer_analysis/2025-09-13_18-40-31_9a4c.parquet" # Adjust if your filename is different
    
    if not os.path.exists(NPZ_FILE) or not os.path.exists(PARQUET_FILE):
        print("[SKIPPED] Audio data files not found.")
        print(f"Please ensure '{NPZ_FILE}' and '{PARQUET_FILE}' exist to run this test.")
    else:
        try:
            # Instantiate the generator
            audio_generator = AudioPromptGenerator(npz_path=NPZ_FILE, parquet_path=PARQUET_FILE)
            
            # Get a batch of prompts
            num_prompts_to_get = 4
            prompt_len_tokens = 256
            
            print(f"\nRequesting {num_prompts_to_get} prompts of length {prompt_len_tokens}...")
            
            prompt_tensor = audio_generator.get_prompts(
                num_prompts=num_prompts_to_get, 
                prompt_length=prompt_len_tokens
            )
            
            # Verify the output
            print("\n[SUCCESS] Generated audio prompt tensor.")
            print(f"  - Shape: {prompt_tensor.shape}")
            print(f"  - Dtype: {prompt_tensor.dtype}")
            
            assert prompt_tensor.shape == (num_prompts_to_get, prompt_len_tokens)
            assert prompt_tensor.dtype == torch.int64
            print("[VERIFIED] Output format is correct.")
            
            # Print a small snippet of the first prompt
            print("\nSnippet of the first prompt (first 2*75 tokens):")
            print(f"  {prompt_tensor[0, :2*75].tolist()}")
            print("\nSnippet of the second prompt (first 2*75 tokens):")
            print(f"  {prompt_tensor[1, :2*75].tolist()}")
            print("\nSnippet of the third prompt (first 2*75 tokens):")
            print(f"  {prompt_tensor[2, :2*75].tolist()}")
            print("\nSnippet of the fourth prompt (first 2*75 tokens):")
            print(f"  {prompt_tensor[3, :2*75].tolist()}")

        except Exception as e:
            print(f"\n[ERROR] An unexpected error occurred during the AudioPromptGenerator test: {e}")
            
    print("\n" + "="*80)
    print("== Self-Tests Complete ==")
    print("="*80)

from encodec import EncodecModel
from mformer_utils import tokenize_audio_on_the_fly, analyze_audio_on_the_fly
from mformer_dataset import Hyperparameters # Import the config class

class OODAudioPromptGenerator:
    """
    Performs on-the-fly tokenization and analysis of an arbitrary audio file
    to generate prompts based on the same structural importance logic used
    in training.
    """
    def __init__(self, ood_audio_path: str, device: str = 'cuda'):
        """
        Initializes the generator by running the full analysis pipeline on the
        provided audio file.

        Args:
            ood_audio_path (str): Path to the out-of-distribution audio file.
            device (str): The device to use for Encodec processing.
        """
        print(f"--- Initializing OOD Audio Prompt Generator for: {ood_audio_path} ---")
        
        # 1. Instantiate analysis components
        params = Hyperparameters()
        encodec_model = EncodecModel.encodec_model_24khz().to(device)
        encodec_model.set_target_bandwidth(6.0) # Match training config

        # 2. Run the on-the-fly pipeline
        tokens, metadata = tokenize_audio_on_the_fly(ood_audio_path, encodec_model)
        
        # NOTE: We use the first codebook stream for length/shape calculations,
        # which is consistent with the IntelligentAudioLoader.
        self.tokens_memmap = tokens[0, :] 
        
        priority_scores, chunk_size_tokens = analyze_audio_on_the_fly(
            ood_audio_path, tokens, metadata['frame_rate'], params
        )

        # 3. Set up internal state to mimic AudioPromptGenerator
        self.weights = torch.from_numpy(priority_scores).float()
        if self.weights.sum() > 0:
            self.weights /= self.weights.sum()
        else:
            self.weights = torch.ones_like(self.weights) / len(self.weights)

        self.total_tokens = len(self.tokens_memmap)
        self.num_chunks = len(self.weights)
        self.chunk_size_tokens = chunk_size_tokens

        print(f"--- OOD Generator Ready. Found {self.total_tokens} tokens across {self.num_chunks} chunks. ---")

    def get_prompts(self, num_prompts: int, prompt_length: int) -> torch.Tensor:
        """
        Samples and returns a batch of prompt sequences from the OOD audio.
        This method is identical to the one in AudioPromptGenerator.
        """
        # (This code is copied directly from the original AudioPromptGenerator.get_prompts)
        if prompt_length >= self.total_tokens:
            raise ValueError(f"prompt_length ({prompt_length}) cannot be larger than total_tokens ({self.total_tokens}).")

        selected_chunk_indices = torch.multinomial(
            self.weights, num_samples=num_prompts, replacement=True
        )
        
        prompts_np = np.zeros((num_prompts, prompt_length), dtype=np.int64)

        for i, chunk_idx in enumerate(selected_chunk_indices):
            start_token = (chunk_idx.item() * self.chunk_size_tokens) - (prompt_length // 2)
            start_token = max(0, start_token)
            end_token = start_token + prompt_length
            
            if end_token > self.total_tokens:
                end_token = self.total_tokens
                start_token = end_token - prompt_length
            
            token_slice = self.tokens_memmap[start_token:end_token]
            prompts_np[i] = token_slice

        return torch.from_numpy(prompts_np)
