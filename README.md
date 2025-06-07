# QuoteFinder

Extract paragraphs matching specific criteria from text files using local language models with multi-model verification.

## Quick Start

```bash
# Install dependencies
pip install beautifulsoup4 ollama

# Pull models (Ollama will auto-start)
ollama pull gemma3:4b qwen3:4b

# Run analysis
python main.py
```

## Configuration

Create a JSON config file:

```json
{
  "model_names": ["some_model", "some_other_model"],
  "target_file": "your_text.html",
  "search_criteria": "specific keyword or phrase"
}
```

## Usage

```bash
python main.py -c config.json              # New analysis with config file
python main.py -d 5                        # Use specific output directory
python main.py -d 5 -r                     # Clear directory and restart
python main.py -v                          # Verbose progress output
python main.py -s                          # Simple mode (quotes only)
python main.py -c config.json -d 5 -v -r   # Combine options
```

## Output Structure

```
output/
├── 0/
│   ├── config.json               # Configuration
│   ├── raw_quotes.txt            # Clean extracted passages
│   ├── detailed_results.txt      # Full analysis with explanations
│   └── progress.json             # Resumption state
└── 1/...
```

## Features

- **Multi-model verification**: All models must agree for higher precision
- **Format support**: HTML (paragraphs) and TXT (stanzas)
- **Resumable**: Interrupt at any time, resume later
- **Organized output**: Numbered directories with progress tracking
