import pandas as pd
import pickle
import os
import gc
from typing import Optional, Callable


class DataLoader:
    def __init__(
        self,
        file_path: str,
        checkpoint_file: str = 'checkpoint.pkl',
        chunk_size: int = 5250,
        checkpoint_interval: int = 50000
    ):
        self.file_path = file_path
        self.checkpoint_file = checkpoint_file
        self.chunk_size = chunk_size
        self.checkpoint_interval = checkpoint_interval
        self.total_rows = 0
        self.combined_df = None

    def load_checkpoint(self) -> bool:
        """Load data from checkpoint if it exists."""
        if not os.path.exists(self.checkpoint_file):
            return False

        print("Found checkpoint file. Loading from checkpoint...")
        try:
            with open(self.checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
                self.combined_df = checkpoint_data['df']
                self.total_rows = checkpoint_data['total_rows']
                print(f"Loaded {self.total_rows} rows from checkpoint")
            return True
        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}")
            return False

    def save_checkpoint(self) -> None:
        """Save current progress to checkpoint file."""
        if self.combined_df is not None:
            print(f"Saving checkpoint at {self.total_rows} rows...")
            with open(self.checkpoint_file, 'wb') as f:
                pickle.dump({
                    'df': self.combined_df,
                    'total_rows': self.total_rows
                }, f)
            gc.collect()

    def process_chunk(self, chunk: pd.DataFrame,
                      chunk_processor: Optional[Callable] = None) -> pd.DataFrame:
        """Process a single chunk of data."""
        if chunk_processor:
            chunk = chunk_processor(chunk)

        self.total_rows += len(chunk)
        print(f"Loaded chunk of {len(chunk)} rows... "
              f"Total rows so far: {self.total_rows}")
        return chunk

    def load_data(self,
                  chunk_processor: Optional[Callable] = None,
                  **read_csv_kwargs) -> pd.DataFrame:
        """Load data in chunks with checkpointing."""
        chunks = []

        # Try to load from checkpoint first
        self.load_checkpoint()

        try:
            for chunk in pd.read_csv(
                self.file_path,
                chunksize=self.chunk_size,
                **read_csv_kwargs
            ):
                try:
                    # Process the chunk
                    processed_chunk = self.process_chunk(
                        chunk, chunk_processor)
                    chunks.append(processed_chunk)

                    # Combine chunks more frequently to manage memory
                    if len(chunks) >= 2:
                        if self.combined_df is None:
                            self.combined_df = pd.concat(
                                chunks, ignore_index=True)
                        else:
                            self.combined_df = pd.concat(
                                [self.combined_df] + chunks,
                                ignore_index=True
                            )
                        chunks = []
                        gc.collect()

                    # Save checkpoint periodically
                    if self.total_rows % self.checkpoint_interval == 0:
                        self.save_checkpoint()

                except Exception as chunk_error:
                    print(f"Error processing chunk: {str(chunk_error)}")
                    continue

            # Combine remaining chunks
            if chunks:
                if self.combined_df is None:
                    self.combined_df = pd.concat(chunks, ignore_index=True)
                else:
                    self.combined_df = pd.concat(
                        [self.combined_df] + chunks,
                        ignore_index=True
                    )

            if self.combined_df is not None:
                print(
                    f"Successfully loaded {len(self.combined_df)} total rows!")
                self.save_checkpoint()  # Save final checkpoint
                return self.combined_df
            else:
                raise ValueError("No data was successfully loaded!")

        except Exception as e:
            print(f"Error loading data: {str(e)}")
            self.save_checkpoint()  # Try to save checkpoint even if there's an error
            raise
