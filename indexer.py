# indexer.py

import os
import json
import numpy as np
import faiss
import torch
from transformers import AutoModel
from PIL import Image

# --- Configuration ---
IMAGE_DIR = "images"
OUTPUT_DIR = "data"
EMBEDDINGS_FILE = os.path.join(OUTPUT_DIR, "image_embeddings.npy")
PATHS_FILE = os.path.join(OUTPUT_DIR, "image_paths.json")
FAISS_INDEX_FILE = os.path.join(OUTPUT_DIR, "faiss.index")

MODEL_NAME = 'jinaai/jina-clip-v2'
BATCH_SIZE = 32  

# --- Main Indexing Logic ---
def main():
    print("Starting the indexing process...")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # --- Step 1: Set up device and load model ---
    device = "cuda" # if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading model '{MODEL_NAME}' from Hugging Face...")
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to(device)

    print("Model loaded successfully.")

    # --- Step 2: Find all image files ---
    print(f"Scanning for images in '{IMAGE_DIR}'...")
    image_paths = []
    for root, _, files in os.walk(IMAGE_DIR):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                image_paths.append(os.path.join(root, file))
    
    if not image_paths:
        print(f"No images found in '{IMAGE_DIR}'. Please add some images and try again.")
        return
    
    print(f"Found {len(image_paths)} images to process.")

    # --- Step 3: Generate embeddings in batches ---
    print(f"Generating embeddings in batches of {BATCH_SIZE}...")
    all_embeddings = []
    successful_paths = []

    for i in range(0, len(image_paths), BATCH_SIZE):
        batch_paths = image_paths[i : i + BATCH_SIZE]
        batch_images = []
        valid_paths_in_batch = []

        for path in batch_paths:
            try:
                img = Image.open(path).convert("RGB")
                batch_images.append(img)
                valid_paths_in_batch.append(path)
            except Exception as e:
                print(f"Warning: Could not load image {path}. Skipping. Error: {e}")
        
        if not batch_images:
            continue
            
        # Get embeddings
        with torch.no_grad(): 
            image_embeds = model.encode_image(batch_images)
        
        all_embeddings.append(image_embeds)
        successful_paths.extend(valid_paths_in_batch)
        print(f"Processed batch {i // BATCH_SIZE + 1}/{(len(image_paths) + BATCH_SIZE - 1) // BATCH_SIZE}...")

    # Concatenate all batch embeddings into a single NumPy array
    all_embeddings = np.vstack(all_embeddings).astype('float32')
    
    # --- Step 4: Save embeddings and file paths ---
    print("Saving embeddings and file paths...")
    np.save(EMBEDDINGS_FILE, all_embeddings)
    with open(PATHS_FILE, 'w') as f:
        json.dump(successful_paths, f)
    print(f"Embeddings saved to {EMBEDDINGS_FILE}")
    print(f"Paths saved to {PATHS_FILE}")

    # --- Step 5: Build and save the FAISS index ---
    if all_embeddings.shape[0] > 0:
        print("Building FAISS index...")
        dimension = all_embeddings.shape[1]
        print(f"Detected embedding dimension: {dimension}") # Will be 768 for this model
        index = faiss.IndexFlatL2(dimension)
        index.add(all_embeddings)

        print("Saving FAISS index...")
        faiss.write_index(index, FAISS_INDEX_FILE)
        print(f"FAISS index saved to {FAISS_INDEX_FILE}")
    else:
        print("No embeddings were generated. Skipping FAISS index creation.")

    print("\nIndexing process completed successfully!")


if __name__ == "__main__":
    main()