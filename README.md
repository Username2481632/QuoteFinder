# PDF Quote Finder

A Python script that processes PDF books paragraph by paragraph using a local language model to identify and extract interesting quotes or passages.

## Setup Instructions for M2 Mac Mini

### 1. Install Ollama

```bash
# Download and install Ollama from https://ollama.ai/download
# Or using homebrew:
brew install ollama

# Start Ollama service
ollama serve
```

### 2. Pull the DeepSeek Model

```bash
# Pull the DeepSeek-R1 distill model (or similar small model)
ollama pull deepseek-r1-distill-llama-8b

# Alternative smaller models if needed:
# ollama pull llama3.2:3b
# ollama pull phi3:mini
```

### 3. Install Python Dependencies

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python pdf_quote_finder.py path/to/your/book.pdf
```

### Advanced Usage

```bash
# Specify output file prefix
python pdf_quote_finder.py book.pdf -o my_quotes

# Use different model
python pdf_quote_finder.py book.pdf -m llama3.2:3b

# Use custom prompt (see custom_prompt_example.txt)
python pdf_quote_finder.py book.pdf --prompt my_custom_prompt.txt
```

## Output

The script generates two output files:

1. **interesting_quotes.json** - Structured data with page numbers and metadata
2. **interesting_quotes.txt** - Human-readable format for easy review

## How It Works

1. **PDF Processing**: Uses `pdfplumber` to extract text while preserving page numbers
2. **Paragraph Splitting**: Intelligently splits text into meaningful paragraphs
3. **LLM Evaluation**: Sends each paragraph to your local model with a prompt asking for "1" (interesting) or "0" (skip)
4. **Logging**: Saves paragraphs rated as "1" along with their page numbers

## Customizing the Prompt

You can customize what the model considers "interesting" by creating a custom prompt file. See `custom_prompt_example.txt` for an example.

## Performance Notes

- Processing time depends on PDF size and model speed
- DeepSeek-R1 distill is optimized for speed while maintaining quality
- On M2 Mac Mini, expect ~1-3 seconds per paragraph depending on model size

## Troubleshooting

### "Model not found" error
```bash
ollama list  # Check available models
ollama pull deepseek-r1-distill-llama-8b  # Pull the model
```

### "Error connecting to Ollama"
```bash
ollama serve  # Make sure Ollama is running
```

### Memory issues with large PDFs
- Use a smaller model like `phi3:mini`
- Process PDFs in smaller sections
- Increase swap space if needed
