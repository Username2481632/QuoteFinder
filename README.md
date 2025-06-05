# Social Comparison Finder

A Python script that processes "The Adventures of Huckleberry Finn" paragraph by paragraph using a local language model to identify passages where characters compare themselves to others.

## What It Finds

The script specifically looks for paragraphs containing:
- Self-comparison thoughts and feelings
- Characters measuring themselves against others
- Expressions of envy, jealousy, or inadequacy
- Competitive internal dialogue
- Social comparison of any kind (appearance, intelligence, status, etc.)
- Self-doubt triggered by seeing others' qualities

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
# Pull the DeepSeek-R1 model
ollama pull deepseek-r1:1.5b
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

### Simple Usage

```bash
python main.py
```

That's it! The script automatically:
- Processes the included Huckleberry Finn text
- Uses the hardcoded social comparison prompt
- Saves results to `social_comparisons.txt`
- Tracks progress in `progress.json`

## Features

### ✅ **Progress Tracking**
- Automatically saves progress after each paragraph
- Resume from where you left off if interrupted
- Progress stored in `progress.json`

### ✅ **Real-time Output**
- Results are appended to `social_comparisons.txt` immediately when found
- No need to wait for the entire process to complete
- See results as they're discovered

### ✅ **Probability-based Evaluation**
- Uses model confidence scores instead of text parsing
- More reliable than simple text matching
- Shows confidence levels for each finding

## Output

The script generates:

1. **social_comparisons.txt** - Real-time results with confidence scores
2. **progress.json** - Progress tracking (deleted when complete)

## How It Works

1. **Text Processing**: Extracts clean text from the included HTML file
2. **Paragraph Splitting**: Intelligently splits text into meaningful paragraphs  
3. **LLM Evaluation**: Uses hardcoded prompt to identify social comparison content
4. **Real-time Logging**: Immediately saves interesting paragraphs as they're found
5. **Progress Tracking**: Allows resuming if interrupted

## Resuming Interrupted Processing

If the script is interrupted (Ctrl+C or system crash), simply run it again:

```bash
python main.py
```

It will automatically resume from where it left off using the `progress.json` file.

## Performance Notes

- Processing time depends on text size and model speed
- DeepSeek-R1 model optimized for speed while maintaining quality
- On M2 Mac Mini, expect ~1-3 seconds per paragraph
- Progress is saved after each paragraph for safety

## Troubleshooting

### "Model not found" error
```bash
ollama list  # Check available models
ollama pull deepseek-r1:1.5b  # Pull the model
```

### "Error connecting to Ollama"
```bash
ollama serve  # Make sure Ollama is running
```

### "Text file not found"
The script expects `huckleberry_finn.html` to be in the same directory. It should have been downloaded automatically when you cloned the repo.
