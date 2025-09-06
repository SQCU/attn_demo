# sampler_utils.py
import random
import pyarrow.parquet as pq
import pandas as pd

class ParquetSampler:
    """
    Provides memory-efficient random sampling of documents from a large
    Parquet file by reading only a subset of row groups.
    """
    def __init__(self, file_path: str):
        print(f"Opening Parquet file for efficient sampling: {file_path}")
        self.pq_file = pq.ParquetFile(file_path)
        self.num_row_groups = self.pq_file.num_row_groups
        if self.num_row_groups == 0:
            raise ValueError("Parquet file has no row groups.")
        print(f"File contains {self.pq_file.metadata.num_rows} documents in {self.num_row_groups} row groups.")

    def get_random_documents(self, num_docs: int, column: str = 'text') -> list[str]:
        """
        Efficiently samples N documents from the Parquet file.
        """
        collected_docs = []
        # Create a shuffled list of row group indices to read from
        row_group_indices = list(range(self.num_row_groups))
        random.shuffle(row_group_indices)

        # Read random row groups until we have enough documents
        for group_index in row_group_indices:
            # Read one row group (a small chunk of the file) into a pandas DataFrame
            # This is the only part that uses significant memory, and it's temporary.
            group_df = self.pq_file.read_row_group(group_index, columns=[column]).to_pandas()
            collected_docs.extend(group_df[column].tolist())
            
            # If we've collected enough, stop reading more of the file
            if len(collected_docs) >= num_docs:
                break
        
        # Return a random sample of the exact size requested from our collection
        if len(collected_docs) < num_docs:
            print(f"Warning: Could only collect {len(collected_docs)} documents, requested {num_docs}.")
            return collected_docs
        else:
            return random.sample(collected_docs, num_docs)