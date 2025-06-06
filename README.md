# QuoteFinder

**Intelligent text analysis using cascaded AI verification**

Extract paragraphs and passages that match specific criteria from any text using local language models with optional multi-model verification for enhanced precision.

## Quick Start

```bash
# Install dependencies
pip install beautifulsoup4 ollama

# Pull AI models (Ollama will auto-start)
ollama pull gemma3:4b qwen3:4b

# Run analysis
python main.py
```

## How It Works

1. **Configure**: Specify your text file, AI models, and search criteria
2. **Process**: AI models analyze each paragraph/stanza for matches
3. **Verify**: Optional multi-model consensus reduces false positives
4. **Extract**: Matching passages saved with explanations and confidence scores

## Configuration

Create a JSON file specifying what to analyze:

```json
{
  "model_names": ["gemma3:4b", "qwen3:4b", "deepseek-r1:8b"],
  "target_file": "your_text.html",
  "search_criteria": "descriptions of character emotions or internal conflict"
}
```

## Key Features

- **Cascaded Verification**: Multiple AI models verify each match for precision
- **Format Agnostic**: Processes HTML (paragraphs) and TXT (stanzas) automatically  
- **Organized Output**: Results saved in numbered directories with progress tracking
- **Resumable**: Interrupted analysis can be resumed from where it left off
- **Real-time Feedback**: Live progress updates with verbose mode
- **Performance Optimized**: Concurrent processing with intelligent model management

## Command Options

```bash
python main.py                    # New analysis in next available directory
python main.py -d 5              # Use specific directory (keeps existing files)  
python main.py -d 5 -r           # Clear directory 5 and restart fresh
python main.py -c my_config.json # Use custom configuration
python main.py -v                # Verbose real-time progress display
python main.py -s                # Simple mode (quotes only, faster)
```

## Output Structure

```
output/
├── 0/
│   ├── raw_quotes.txt           # Clean extracted passages
│   ├── detailed_results.txt     # Full analysis with explanations  
│   └── progress.json           # Resumption state
└── 1/...
```

## Multi-Model Verification

- **Single Model**: `["qwen3:4b"]` → Fast standard analysis
- **Dual Verification**: `["gemma3:4b", "qwen3:4b"]` → Enhanced accuracy  
- **Cascade Verification**: `["model1", "model2", "model3"]` → Maximum precision

Each passage must be approved by **all** specified models to be included in results, dramatically reducing false positives while maintaining sensitivity.

---

*Built for researchers, analysts, and anyone who needs reliable automated text analysis with explainable AI reasoning.*
