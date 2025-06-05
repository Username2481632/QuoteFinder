# Social Comparison Finder

Extract paragraphs containing social comparisons from text using a local language model. Supports both HTML files (like Huckleberry Finn) and TXT files (like The Odyssey).

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

# Use different configuration files
python main.py -c sample_config.json      # Default configuration
python main.py -c my_custom_config.json   # Your custom analysis

# Simple mode - raw quotes only (faster)
python main.py --simple
python main.py -s

# Verbose output
python main.py -v

# Combine flags
python main.py -d 3 -s -v -c sample_config.json
```

## File Format Support

The script automatically detects file types and processes them appropriately:

- **HTML files** (`.html`): Split into paragraphs for prose analysis
- **TXT files** (`.txt`): Split into stanzas for poetry analysis

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

Configuration files specify the model, target file, and analysis prompts:

```json
{
  "model_name": "qwen3:4b",
  "target_file": "huckleberry_finn.html",
  "classification_prompt": "Your analysis prompt here...",
  "explanation_prompt": "Your explanation prompt here..."
}
```

### Sample Configuration

The included `sample_config.json` provides a starting template:

```json
{
  "model_name": "qwen3:4b",
  "target_file": "huckleberry_finn.html",
  "classification_prompt": "Your analysis prompt here...",
  "explanation_prompt": "Your explanation prompt here..."
}
```

### Creating Custom Configs

1. Copy `sample_config.json` to a new file (e.g., `my_analysis.json`)
2. Update `target_file` to point to your text file 
3. Modify prompts for your specific analysis needs
4. Ensure your target file is in the project directory

### File Types Supported:
- **HTML files** (`.html`): Split into paragraphs for prose analysis
- **TXT files** (`.txt`): Split into stanzas for poetry analysis

## Features

- **Automatic Ollama startup** - Starts Ollama if not running
- **Multiple file formats** - HTML (paragraphs) and TXT (stanzas)
- **Numbered run directories** - Organize multiple experimental runs
- **Progress tracking** - Resume interrupted processing
- **Real-time output** - See results as they're found
- **Configurable** - Easy prompt and model customization
- **Cross-platform** - Works on Windows, Mac, Linux
