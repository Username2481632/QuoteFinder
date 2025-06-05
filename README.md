# Social Comparison Finder

Extract paragraphs containing social comparisons from text using a local language model. Supports both HTML files (like Huckleberry Finn) and TXT files (like The Odyssey).

## Setup

1. **Install Ollama** from https://ollama.ai/download
2. **Pull a model**: `ollama pull qwen3:4b`
3. **Install dependencies**: `pip install beautifulsoup4 ollama`

## Usage

```bash
# Create new directory with next available number (default)
python main.py

# Use specific directory number (keeps existing files)
python main.py --directory 5
python main.py -d 0

# Clear files in specific directory and restart
python main.py --directory 5 --restart
python main.py -d 0 -r

# Use different configuration files
python main.py -c sample_config.json      # Default configuration
python main.py -c my_custom_config.json   # Your custom analysis

# Simple mode - raw quotes only (faster)
python main.py --simple
python main.py -s

# Verbose output
python main.py -v

# Combine flags
python main.py -d 3 -r -s -v -c sample_config.json
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

- **No flags**: Create next available numbered directory (e.g., if 0,1,2 exist, creates 3)
- **`-d N/--directory N`**: Use directory `N` (keeps existing files)
- **`-r/--restart`**: Clear files in whichever directory gets used (requires `-d`)
- **`-d N -r`**: Use directory `N` and clear existing files to restart fresh

## Configuration

Configuration files specify the model, target file, and search criteria:

```json
{
  "model_name": "qwen3:4b",
  "target_file": "huckleberry_finn.html",
  "search_criteria": "Your search description here..."
}
```

### Sample Configuration

The included `sample_config.json` provides a starting template:

```json
{
  "model_name": "qwen3:4b",
  "target_file": "huckleberry_finn.html",
  "search_criteria": "explicit description of a character comparing themselves to another person"
}
```

### Creating Custom Configs

1. Copy `sample_config.json` to a new file (e.g., `my_analysis.json`)
2. Update `target_file` to point to your text file 
3. Modify `search_criteria` to describe what you're looking for
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
