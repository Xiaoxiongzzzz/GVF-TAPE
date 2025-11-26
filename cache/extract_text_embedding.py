import h5py
import os
import numpy as np
from tqdm import tqdm
import torch


def extract_and_combine_embeddings(input_dir, output_file):
    """
    Extract text embeddings from multiple hdf5 files and combine them into one file

    Args:
        input_dir (str): Directory containing hdf5 files
        output_file (str): Path to save the combined embeddings
    """
    # Get all hdf5 files
    hdf5_files = [f for f in os.listdir(input_dir) if f.endswith(".hdf5")]
    print(f"Found {len(hdf5_files)} hdf5 files in {input_dir}")

    # Dictionary to store all embeddings
    all_embeddings = {}

    # Process each file
    for filename in tqdm(hdf5_files, desc="Processing files"):
        file_path = os.path.join(input_dir, filename)

        with h5py.File(file_path, "r") as f:
            # Extract text and embedding
            text = filename.split(".")[0].replace("_demo", "")
            embedding = f["text_embed"][:]

            # Store in dictionary
            all_embeddings[text] = embedding

    print(f"\nSuccessfully processed {len(all_embeddings)} embeddings")

    # Save combined embeddings
    print(f"Saving combined embeddings to {output_file}")
    with h5py.File(output_file, "w") as f:
        # Create groups for texts and embeddings
        embedding_group = f.create_group("embeddings")

        # Store the data
        for i, (text, embedding) in enumerate(all_embeddings.items()):
            embedding_group.create_dataset(text, data=embedding)

        # Store metadata
        f.attrs["num_embeddings"] = len(all_embeddings)

    print("Done!")


def main():
    # Configure paths
    input_dir = "/mnt/data0/xiaoxiong/atm_libero/libero_spatial"  # 修改为你的输入目录
    output_file = "./cache/libero_spatial_embedding.hdf5"  # 修改为你想要保存的位置

    # Extract and combine embeddings
    extract_and_combine_embeddings(input_dir, output_file)

    # Verify the saved file
    with h5py.File(output_file, "r") as f:
        num_embeddings = f.attrs["num_embeddings"]
        print(f"\nVerification:")
        print(f"Total number of embeddings saved: {num_embeddings}")
        print(f"Available groups: {list(f.keys())}")


if __name__ == "__main__":
    main()
