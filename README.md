# Social Comparison Finder

Extract paragraphs containing social comparisons from text using a local language model.

## Setup

1. **Install Ollama** from https://ollama.ai/download
2. **Pull a model**: `ollama pull qwen3:4b`
3. **Install dependencies**: `pip install beautifulsoup4 ollama`

## Usage

```bash
# Process with explanations (creates detailed_results.txt + raw_quotes.txt)
python main.py

# Simple mode - raw quotes only (faster)
python main.py --simple
python main.py -s

# Verbose output
python main.py -v

# Custom config
python main.py --config my_config.json
```

## Configuration

Edit `sample_config.json` to change the model or prompts:

```json
{
  "model_name": "qwen3:4b",
  "classification_prompt": "Your prompt here...",
  "explanation_prompt": "Your explanation prompt here..."
}
```

## Output

- `./output/raw_quotes.txt` - Clean quotes: `"paragraph1"\n"paragraph2"`
- `./output/detailed_results.txt` - Full results with explanations (unless `--simple`)

## Features

- **Progress tracking** - Resume interrupted processing
- **Real-time output** - See results as they're found
- **Configurable** - Easy prompt and model customization
- **Cross-platform** - Works on Windows, Mac, Linux
