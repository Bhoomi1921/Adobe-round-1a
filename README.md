PDF Structure Analyzer Pro

An advanced PDF document structure extraction tool powered by machine learning and natural language processing.

## Features

### 🎯 Core Capabilities
- **Smart Heading Detection**: AI-powered identification of document headings and titles
- **Hierarchical Structure Building**: Multi-level document outline generation
- **Semantic Analysis**: Context-aware content classification using transformer embeddings
- **Confidence Scoring**: Reliability metrics for each extracted element
- **Multi-Modal Classification**: Combines visual, textual, and semantic signals
- **Multi-Lingual**:can give outputs even for another language pdfs like japanese, chinese etc.

### 🧠 Advanced Analysis
- **Adaptive Clustering**: Dynamic hierarchy level detection based on document characteristics
- **Title Extraction**: Intelligent document title identification
- **Content Type Classification**: Distinguishes between headings, content, titles, and references
- **Position-Aware Processing**: Considers document layout and positioning

### 💾 Export Options
- **Multiple Formats**: JSON, text outlines, batch summaries
- **ZIP Archives**: Bulk download of all results
- **Individual Files**: Separate results for each processed document

## Multilingual Support
The system supports document processing in multiple languages including:
- Japanese (日本語)
- Chinese (中文)
- Korean (한국어)
- European languages (French, Spanish, German, etc.)

## Installation

### Prerequisites
- Python 3.8 or higher
- ONNX model files (MiniLM-L6-H384)
- Required Python packages (see requirements.txt)

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd pdf-structure-analyzer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Setup model files**
   ```
   model/
   ├── model.onnx
   └── tokenizer/
       ├── config.json
       ├── special_tokens_map.json
       ├── tokenizer_config.json
       ├── tokenizer.json
       └── vocab.txt
   ```

4. **Create directories**
   ```bash
   mkdir -p documents results
   ```

## Usage

### Command Line Interface
```bash
# Process all PDFs in documents directory
python main.py

# Process with custom configuration
python main.py --config custom_config.json
```

### Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up --build

# Access at http://localhost:8501
```

## Configuration

Edit `config.json` to customize analysis parameters:

```json
{
    "analysis_settings": {
        "confidence_threshold": 0.7,
        "max_hierarchy_levels": 6,
        "enable_semantic_analysis": true
    },
    "clustering": {
        "font_size_weight": 0.4,
        "semantic_weight": 0.3,
        "position_weight": 0.2
    }
}
```

## Output Format

### JSON Structure
```json
{
    "document_title": "Document Title",
    "analysis_timestamp": "2024-01-01T12:00:00",
    "document_outline": [
        {
            "hierarchy_level": "H1",
            "content": "Main Heading",
            "page": 1,
            "confidence_score": 0.95,
            "coordinates": {"x": 100, "y": 200},
            "font_properties": {...},
            "semantic_relevance": 0.88
        }
    ],
    "analytics": {
        "total_sections_found": 15,
        "pages_analyzed": 10,
        "average_confidence": 0.82
    }
}
```

## Architecture

### Core Components

1. **DocumentParser**: Extracts text elements with formatting information
2. **SemanticAnalyzer**: Generates embeddings and analyzes semantic relationships
3. **ContentClassifier**: Classifies elements and calculates probabilities
4. **HierarchyBuilder**: Constructs document hierarchy using multi-modal clustering
5. **PDFDocumentAnalyzer**: Main orchestrator coordinating all components

### Key Innovations

- **Multi-Modal Feature Vectors**: Combines semantic embeddings with visual features
- **Adaptive Clustering**: Dynamically determines optimal number of hierarchy levels
- **Confidence Scoring**: Multiple metrics for reliability assessment
- **Context-Aware Classification**: Considers surrounding elements for better accuracy

## Performance

- **Processing Speed**: ~2-5 seconds per page depending on complexity
- **Memory Usage**: ~200-500MB per document depending on size
- **Accuracy**: >90% for well-formatted documents with clear structure

## Troubleshooting

### Common Issues

1. **Model files not found**
   - Ensure ONNX model and tokenizer files are in the correct locations
   - Check file permissions

2. **Out of memory errors**
   - Reduce batch size in configuration
   - Process fewer documents simultaneously

3. **Low accuracy results**
   - Adjust confidence thresholds
   - Enable semantic analysis for better results
   - Check if document has clear visual hierarchy

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Hugging Face Transformers for the embedding model
- PyMuPDF for PDF processing capabilities
- Streamlit for the interactive web interface
- scikit-learn for clustering algorithms
