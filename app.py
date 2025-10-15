import gradio as gr
import faiss
import numpy as np
import json
import torch
from transformers import AutoModel
from PIL import Image

# --- Configuration ---
DATA_DIR = "data"
FAISS_INDEX_FILE = f"{DATA_DIR}/faiss.index"
PATHS_FILE = f"{DATA_DIR}/image_paths.json"
MODEL_NAME = 'jinaai/jina-clip-v2'
NUM_RESULTS = 4

# --- Load Resources ---
try:
    print("Loading FAISS index...")
    index = faiss.read_index(FAISS_INDEX_FILE)

    print("Loading image paths...")
    with open(PATHS_FILE, 'r') as f:
        image_paths = json.load(f)

    # Set up device and load model
    device = "cuda" # if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print(f"Loading model '{MODEL_NAME}' from Hugging Face...")
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to(device)
    print("Resources loaded successfully.")

except FileNotFoundError:
    print("=" * 50)
    print("ERROR: Index files not found!")
    print(f"Please delete any old 'data' directory and run 'indexer.py' first.")
    print("=" * 50)
    exit()

# --- Helper function to resize images for display ---
def resize_for_display(image_path, max_size=512):
    """Resize image to fit within max_size for display while maintaining aspect ratio"""
    try:
        image = Image.open(image_path)
        
        # Get current dimensions
        width, height = image.size
        
        # Calculate scaling factor
        if width > max_size or height > max_size:
            scale = min(max_size / width, max_size / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # Resize with high quality resampling
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return image
    except Exception as e:
        print(f"Error resizing image {image_path}: {e}")
        return None

# --- Core Search Logic ---
def search(query_image, query_text):
    if query_image is None and (query_text is None or query_text.strip() == ""):
        return []

    query_vector = None
    
    # --- Step 1: Encode the query to get a vector ---
    if query_image is not None:
        with torch.no_grad():
            query_vector = model.encode_image([query_image])
            
    elif query_text is not None and query_text.strip() != "":
        with torch.no_grad():
            query_vector = model.encode_text([query_text])

    if query_vector is None:
        return []

    # FAISS expects a 2D array of float32
    query_vector = query_vector.astype('float32')

    # --- Step 2: Search the FAISS index ---
    distances, indices = index.search(query_vector, k=NUM_RESULTS)

    # --- Step 3: Map indices back to file paths and resize for display ---
    results = []
    for i in indices[0]:
        image_path = image_paths[i]
        resized_image = resize_for_display(image_path)
        if resized_image is not None:
            results.append(resized_image)
    
    return results

# --- Gradio UI ---
with gr.Blocks(theme=gr.themes.Soft(), title="Multimodal Photo Search") as demo:
    gr.Markdown("# 🧠 Multimodal Photo Search Engine")
    gr.Markdown(
        "Search your personal photo library using an example image OR a text description. "
    )
    
    with gr.Tabs():
        with gr.TabItem("🖼️ Search by Image"):
            image_input = gr.Image(type="pil", label="Drop an image here")
        
        with gr.TabItem("✍️ Search by Text"):
            text_input = gr.Textbox(label="Enter a text description", placeholder="e.g., A photo of a dog playing in a park...")

    search_button = gr.Button("Search", variant="primary")

    gallery = gr.Gallery(
        label="Search Results",
        show_label=True,
        columns=4,
        object_fit="contain",
        height="auto"
    )

    # --- Connect UI components to the search function ---
    text_input.submit(fn=search, inputs=[image_input, text_input], outputs=gallery, show_progress="hidden")
    image_input.change(fn=search, inputs=[image_input, text_input], outputs=gallery, show_progress="hidden")
    search_button.click(fn=search, inputs=[image_input, text_input], outputs=gallery, show_progress="hidden")

if __name__ == "__main__":
    print("Launching Gradio App...")
    demo.launch(share=True)