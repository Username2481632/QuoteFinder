# Quote Finder

Extract paragraphs containing specific content from text using multiple local language models with cascade verification. Supports both HTML files (like Huckleberry Finn) and TXT files (like The Odyssey).

## Multi-Model Cascade Verification

This tool uses a **cascade verification system** where multiple AI models work together to ensure high-precision results:

1. **First model** scans text and identifies potential matches
2. **Subsequent models** verify each positive detection
3. **Any negative vote** → immediate rejection
4. **All positive votes** → approved match with combined explanations

This dramatically reduces false positives while maintaining high recall.

## Setup

1. **Install Ollama** from https://ollama.ai/download
2. **Pull models**: 
   ```bash
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

- **`model_names`**: Array of models for cascade verification
- **First model**: Primary detector (fast, high-recall model recommended)
- **Verification models**: Secondary validators (accuracy-focused models)
- **Minimum**: 1 model (standard classification)
- **Recommended**: 2-3 models for optimal precision/recall balance

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

- **Cascade verification** - Multiple AI models validate each match for high precision
- **Automatic model management** - Models load on-demand and stay cached for performance
- **Automatic Ollama startup** - Starts Ollama if not running
- **Multiple file formats** - HTML (paragraphs) and TXT (stanzas)
- **Numbered run directories** - Organize multiple experimental runs
- **Progress tracking** - Resume interrupted processing
- **Graceful cancellation** - Ctrl+C stops after current batch, saves progress
- **Real-time output** - See results as they're found
- **Configurable** - Easy prompt and model customization
- **Performance optimized** - Model keep-alive and parallel batch processing
- **Combined analysis** - Classification and explanation in single model call
- **Cross-platform** - Works on Windows, Mac, Linux

## Model Recommendations

### Fast Detection + High Precision
```json
{
  "model_names": ["gemma3:4b", "qwen3:4b"]
}
```

### Maximum Precision (3-model consensus)
```json
{
  "model_names": ["gemma3:4b", "qwen3:4b", "deepseek-r1:8b"]
}
```

### Single Model (fastest)
```json
{
  "model_names": ["qwen3:4b"]
}
```

## How Cascade Verification Works

### Example with 3 Models: `["gemma3:4b", "qwen3:4b", "deepseek-r1:8b"]`

1. **Step 1**: `gemma3:4b` analyzes each paragraph
   - If it says "0" (no match) → paragraph rejected immediately
   - If it says "1" (match) → proceed to verification

2. **Step 2**: `qwen3:4b` verifies the potential match
   - If it says "0" → paragraph rejected (cascade broken)
   - If it says "1" → proceed to final verification

3. **Step 3**: `deepseek-r1:8b` final verification
   - If it says "0" → paragraph rejected
   - If it says "1" → ✅ **APPROVED** (all models agree)

### Output Example
```
  Model 1 (gemma3:4b): 1 (conf: 0.95)
  Model 2 (qwen3:4b): 1 (conf: 0.89)  
  Model 3 (deepseek-r1:8b): 1 (conf: 0.92)
  Cascade approved by all 3 models
● found (conf: 0.95)
```

### Benefits
- **Eliminates false positives**: All models must agree
- **Maintains recall**: Only one model needs to detect initially
- **Rich explanations**: Combines insights from multiple models
- **Flexible**: Use 1-5+ models based on precision needs
