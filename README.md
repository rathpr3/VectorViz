# 🔍 Embedding Explorer

A web application that explores semantic relationships between words using AI embeddings. Built with FastAPI backend and interactive HTML/JavaScript frontend.

## Features

- **Text Input**: Enter comma-separated words or short phrases
- **AI Embeddings**: Generate embeddings using HuggingFace sentence-transformers
- **Similarity Analysis**: Compute pairwise cosine similarities between all embeddings
- **Interactive Visualizations**: Three different views of the data:
  - **Similarity Heatmap**: Color-coded matrix showing relationships between all words
  - **Word Similarities**: Bar chart comparing one word to all others
  - **2D Embeddings**: PCA-reduced scatter plot for spatial visualization

## Architecture

- **Backend**: FastAPI with Python
- **Frontend**: HTML/CSS/JavaScript with Plotly.js for charts
- **ML**: HuggingFace sentence-transformers for embeddings
- **Visualization**: Plotly.js for interactive charts
- **No Database**: All processing done in memory

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Application

```bash
uvicorn main:app --reload
```

The application will be available at `http://localhost:8000`

### 3. Access the Web Interface

Open your browser and navigate to `http://localhost:8000`

## Usage

1. **Enter Words**: Type a list of words separated by commas in the input box
   - Example: `cat, dog, bird, fish, computer, technology, love, happiness, sadness, anger`

2. **Analyze**: Click the "Analyze" button to generate embeddings and similarities

3. **Explore Results**: Use the three tabs to view different visualizations:
   - **Similarity Heatmap**: See how all words relate to each other
   - **Word Similarities**: Select a word to see its similarity to others
   - **2D Embeddings**: View words in 2D space based on semantic similarity

## API Endpoints

### POST `/embeddings`

Generates embeddings and computes similarities for a list of words.

**Request Body:**
```json
{
  "words": ["cat", "dog", "bird", "fish"]
}
```

**Response:**
```json
{
  "words": ["cat", "dog", "bird", "fish"],
  "embeddings": [[...], [...], [...], [...]],
  "similarity_matrix": [[1.0, 0.8, 0.6, 0.5], ...],
  "embeddings_2d": [[x1, y1], [x2, y2], ...]
}
```

## Technical Details

- **Model**: Uses `all-MiniLM-L6-v2` from sentence-transformers (384-dimensional embeddings)
- **Similarity**: Cosine similarity between normalized embedding vectors
- **Dimensionality Reduction**: PCA to 2D for visualization
- **Frontend**: Responsive design with modern CSS and interactive JavaScript

## Example Use Cases

- **Semantic Analysis**: Understand how words relate semantically
- **Concept Clustering**: Group related concepts together
- **Language Learning**: Explore relationships between words in different languages
- **Content Analysis**: Analyze similarity between documents or phrases

## Development

### Project Structure
```
VectorViz/
├── main.py              # FastAPI backend
├── index.html           # Frontend HTML/JS
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

### Running in Development Mode

```bash
# Install dependencies
pip install -r requirements.txt

# Run with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Or run directly
python main.py
```

### Customization

- **Model**: Change the sentence transformer model in `main.py`
- **Visualizations**: Modify chart layouts and styles in `index.html`
- **Styling**: Update CSS in the `<style>` section of `index.html`

## Troubleshooting

### Common Issues

1. **Model Download**: First run may take time to download the sentence transformer model
2. **Memory**: Large word lists may require more memory
3. **Dependencies**: Ensure all packages are installed correctly

### Error Messages

- **"Model not loaded"**: Check if sentence-transformers installed correctly
- **"At least 2 words required"**: Ensure input contains multiple words
- **"Error processing embeddings"**: Check word format and try again

## License

This project is open source and available under the MIT License.

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the application!
