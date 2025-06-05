# Social Comparison Finder

Extract paragraphs containing social comparisons from text using a local language model.

## Setup

1. **Install Ollama** from https://ollama.ai/download
2. **Pull a model**: `ollama pull qwen3:4b`
3. **Install dependencies**: `pip install beautifulsoup4 ollama`

## Usage

```bash
# Continue latest run (or start from directory 0 if none exist)
python main.py

# Start fresh run in next available directory
python main.py --restart
python main.py -r

# Use specific directory number (clears existing files)
python main.py --directory 5
python main.py -d 0

# Simple mode - raw quotes only (faster)
python main.py --simple
python main.py -s

# Verbose output
python main.py -v

# Custom config
python main.py --config my_config.json

# Combine flags
python main.py -d 3 -s -v -c thesis_config.json
```

## Directory Management

The script organizes results in numbered subdirectories:

```
output/
├── 0/
│   ├── raw_quotes.txt
│   ├── detailed_results.txt
│   └── progress.json
├── 1/
│   ├── raw_quotes.txt
│   ├── detailed_results.txt
│   └── progress.json
└── 2/
    └── ...
```

- **No flags**: Continue with highest numbered directory (or create `0/` if none exist)
- **`-r/--restart`**: Create next available numbered directory
- **`-d N/--directory N`**: Use directory `N` (clears existing files to restart)

## Configuration

Edit `sample_config.json` to change the model or prompts:

```json
{
  "model_name": "qwen3:4b",
  "classification_prompt": "Your prompt here...",
  "explanation_prompt": "Your explanation prompt here..."
}
```

## Features

- **Automatic Ollama startup** - Starts Ollama if not running
- **Numbered run directories** - Organize multiple experimental runs
- **Progress tracking** - Resume interrupted processing
- **Real-time output** - See results as they're found
- **Configurable** - Easy prompt and model customization
- **Cross-platform** - Works on Windows, Mac, Linux
