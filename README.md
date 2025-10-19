# Multimodal AI Photo Search Engine

This project allows you to build and run your own personal photo search engine. You can search your entire photo library using either an example image (image-to-image search) or a natural language description (text-to-image search).

This is the code for the YouTube video: https://youtu.be/sr-lgOLKYAM

## Features

-   **Multimodal Search**: Find photos using text or other photos.
-   **High-Speed**: Uses FAISS for near-instant search, even with tens of thousands of images.
-   **State-of-the-Art AI**: Powered by Jina AI's v2 CLIP model for understanding the semantic content of your queries.
-   **Simple Web UI**: An easy-to-use interface built with Gradio.

## How It Works

1.  **Indexing (`indexer.py`)**:
    -   The script scans a directory of your photos.
    -   For each photo, it uses the Jina AI model to generate a numerical representation (an "embedding" or "vector") that captures its meaning.
    -   All these vectors are stored in a highly efficient FAISS index, which is like a searchable map of your photos' meanings.

2.  **Application (`app.py`)**:
    -   A Gradio web server is launched.
    -   When you provide a query (text or image), it's converted into an embedding vector using the same Jina model.
    -   FAISS then uses this query vector to find the most similar image vectors in its index almost instantly.
    -   The corresponding images are then displayed in the gallery.

## Setup and Usage

### Step 1: Clone the Repository

```bash
git clone https://github.com/Zaharah/multimodal_ai_photo_search
cd multimodal-photo-search
```

### Step 2: Set Up a Virtual Environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### Step 3: Install Dependencies

Install all the required Python libraries using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### Step 4: Add Your Photos

Create a folder named `images` in the project directory and copy your personal photos into it. You can also organize them in subfolders.

```
multimodal-photo-search/
├── images/
│   ├── vacations/
│   │   ├── beach.jpg
│   │   └── mountains.png
│   └── family/
│       └── gathering.jpg
└── ...
```

### Step 5: Run the Indexer

This is a **one-time process** to analyze your photos and build the search index. Depending on the number of photos and your computer's speed/computing power, this might take some time.

```bash
python indexer.py
```

This will create a `data/` directory containing `image_embeddings.npy`, `image_paths.json`, and `faiss.index`.

### Step 6: Launch the Application

Now, run the main application to start the web UI.

```bash
python app.py
```

Open the URL printed in your terminal (usually `http://127.0.0.1:7860`) in your web browser to start searching!


