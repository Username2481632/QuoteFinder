# Quote Finder

Extract paragraphs containing specific content from text using local language models. Supports both HTML files (like Huckleberry Finn) and TXT files (like The Odyssey).

## Multi-Model Support

The tool supports using multiple AI models for improved accuracy:

- **Single model**: Fast, standard classification
- **Multiple models**: Enhanced precision through verification
- **Flexible configuration**: Use 1-5+ models based on your needs

## Setup

1. **Install Ollama** from https://ollama.ai/download
2. **Pull models**: 
   ```bash
   # Example
   ollama pull gemma3:4b
   ollama pull qwen3:4b
   ollama pull deepseek-r1:8b
   ```
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

Configuration files specify the models, target file, and search criteria:

```json
{
  "model_names": ["gemma3:4b", "qwen3:4b", "deepseek-r1:8b"],
  "target_file": "huckleberry_finn.html",
  "search_criteria": "Your search description here..."
}
```

### Multi-Model Setup

- **`model_names`**: Array of models to use
- **Single model**: `["qwen3:4b"]` - Fast, standard classification
- **Multiple models**: `["gemma3:4b", "qwen3:4b"]` - Enhanced accuracy through verification
- **Custom setup**: Use any combination of available models

### Sample Configuration

The included `sample_config.json` provides a starting template:

```json
{
  "model_names": ["gemma3:4b", "qwen3:4b", "deepseek-r1:8b"],
  "target_file": "huckleberry_finn.html",
  "search_criteria": "explicit description of a character comparing themselves to another person"
}
```

### Creating Custom Configs

1. Copy `sample_config.json` to a new file (e.g., `my_analysis.json`)
2. Update `target_file` to point to your text file 
3. Update `model_names` array with your preferred models
4. Modify `search_criteria` to describe what you're looking for
5. Ensure your target file is in the project directory

### File Types Supported:
- **HTML files** (`.html`): Split into paragraphs for prose analysis
- **TXT files** (`.txt`): Split into stanzas for poetry analysis

## Features

- **Multi-model support** - Use single or multiple AI models for classification
- **Automatic model management** - Models load on-demand and stay cached for performance
- **Automatic Ollama startup** - Starts Ollama if not running
- **Multiple file formats** - HTML (paragraphs) and TXT (stanzas)
- **Numbered run directories** - Organize multiple experimental runs
- **Progress tracking** - Resume interrupted processing
- **Graceful cancellation** - Ctrl+C stops after current batch, saves progress
- **Real-time output** - See results as they're found
- **Configurable** - Easy prompt and model customization
- **Performance optimized** - Model keep-alive and parallel batch processing
- **Cross-platform** - Works on Windows, Mac, Linux

## Model Setup Options

### Single Model (fastest)
```json
{
  "model_names": ["qwen3:4b"]
}
```

### Dual Model Verification
```json
{
  "model_names": ["gemma3:4b", "qwen3:4b"]
}
```

### Enhanced Precision
```json
{
  "model_names": ["gemma3:4b", "qwen3:4b", "deepseek-r1:8b"]
}
```

## Multi-Model Verification

When using multiple models, each paragraph is processed sequentially:

1. **First model** analyzes the text
2. **Additional models** verify positive results
3. **All models must agree** for a match to be approved

### Example Output
```
  Model 1 (gemma3:4b): 1 (conf: 0.95)
  Model 2 (qwen3:4b): 1 (conf: 0.89)  
  ✓ Match approved by all models
● found (conf: 0.95)
```

This approach reduces false positives while maintaining good detection rates.
