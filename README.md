# Financial Document Analysis Chatbot

A smart chatbot built with LangChain, Groq, and Streamlit that analyzes financial documents (like fund factsheets) and provides a accurate data.

## Features

-  **Advanced PDF Processing**
  - Extracts text, tables, and structured data from PDFs
  - Uses multiple loaders (PDFMiner + Unstructured) for comprehensive extraction
  - Preserves document layout and formatting

-  **Intelligent Question Answering**
  - RAG (Retrieval Augmented Generation) based responses
  - Context-aware answers from document content
  - Supports complex financial calculations and comparisons

-  **Smart Features**
  - FAISS vector store for efficient document retrieval
  - Chunking with overlap for context preservation
  - Sentence transformer embeddings for semantic search

## Installation

1. Clone the repository and create a virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install the required packages:
   ```powershell
   pip install -e .
   ```
   Or install dependencies directly:
   ```powershell
   pip install langchain-groq langchain-community streamlit plotly numpy pandas python-dotenv
   ```

3. Set up your environment variables:
   ```
   GROQ_API_KEY=your_groq_api_key
   ```

## Usage

1. Start the chatbot:
   ```powershell
   streamlit run bajaj_finserv.py
   ```

2. Upload or use an existing financial document (PDF)

3. Ask questions about the document, such as:
   - "List the top 5 holdings with their weights"
   - "Compare fund performance across different time periods"
   - "Show me the asset allocation breakdown"
   - "What are the risk metrics like Sharpe ratio?"

4. The chatbot will:
   - Provide detailed text responses
   - Display interactive tables for structured data
   - Show charts for numeric comparisons

## Example Questions

### For Tables and Lists
- "List top 5 holdings of the Consumption Fund with weights"
- "Compare the allocation between equity and debt"
- "State the YTM, Macaulay Duration and Average Maturity for the Money Market Fund"

## Technical Details

### Components
- **LangChain**: Framework for building LLM applications
- **Groq**: Large Language Model provider
- **FAISS**: Vector store for efficient similarity search
- **Streamlit**: Web interface and visualization
- **Plotly**: Interactive charts and graphs
- **PDFMiner + Unstructured**: PDF processing and data extraction

### Document Processing
- Chunks size: 2000 characters
- Chunk overlap: 400 characters
- Embedding model: all-MiniLM-L12-v2
- Retrieval method: Similarity search with k=10

## Project Structure
```
.
├── bajaj_finserv.py     # Main application file
├── pyproject.toml       # Project configuration
├── .env                 # Environment variables (create this)
└── README.md           # This documentation
```

## Development

To extend the chatbot's capabilities:

1. **Add New Visualizations**:
   - Implement new chart types in `visualize_data()`
   - Add custom plotly configurations
   - Support more data formats

2. **Enhance Document Processing**:
   - Add support for more document types
   - Implement additional data extractors
   - Optimize chunking parameters

3. **Improve Analysis**:
   - Add financial calculation functions
   - Implement comparative analysis
   - Add historical data tracking

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- LangChain for the RAG framework
- Groq for the LLM API
- Streamlit for the web interface

- The open-source community for various tools and libraries
