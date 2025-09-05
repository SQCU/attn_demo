# prompt_utils.py
import random
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
        if file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
            # Assuming the text is in a column named 'text'
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
        if num_prompts > len(self.documents):
            print(f"Warning: Requesting more prompts ({num_prompts}) than available documents ({len(self.documents)}). Returning all documents.")
            selected_docs = self.documents
        else:
            selected_docs = random.sample(self.documents, num_prompts)
        
        # Truncate each document to the desired prompt length
        prompts = [doc[:prompt_length] for doc in selected_docs]
        return prompts