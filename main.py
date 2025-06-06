#!/usr/bin/env python3
"""
Quote Finder - Extract social comparison paragraphs from text using a local LLM

This script processes HTML/text files paragraph by paragraph, sends each paragraph
to a local language model via Ollama, and logs paragraphs that show people
comparing themselves to others.

Requirements:
- Ollama installed and running
- DeepSeek-R1 distill model pulled in Ollama
- Python packages: beautifulsoup4, ollama

Usage:
    python main.py [--verbose]
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
import ollama


class TextClassifier:
    def __init__(self, config_file: str = 'sample_config.json', verbose: bool = False, simple: bool = False, restart: bool = False, directory: Union[None, int] = None):
        """
        Initialize the Text Classifier.
        
        Args:
            config_file: Path to the configuration JSON file
            verbose: Whether to show detailed debug output
            simple: If True, only generate raw quotes without explanations
            restart: If True, start fresh with next available directory number
            directory: If provided, use this specific directory number
        """
        self.config = self._load_config(config_file)
        self.model_name = self.config['model_name']
        self.verbose = verbose
        self.simple = simple
        self.restart = restart
        self.directory = directory
        self.prompt_template = self.config['search_criteria']
        self.search_criteria = self.config['search_criteria']
        
        # Progress tracking
        self.start_time = None
        
        # Create main output directory
        self.base_output_dir = Path("output")
        self.base_output_dir.mkdir(exist_ok=True)
        
        # Determine the run directory
        self.output_dir = self._get_run_directory()
        self.output_dir.mkdir(exist_ok=True)
        
        # Set output files (all within the run directory)
        self.progress_file = self.output_dir / "progress.json"
        self.raw_quotes_file = self.output_dir / "raw_quotes.txt"
        self.detailed_results_file = self.output_dir / "detailed_results.txt"
        
        self.results_count = 0
        self.positive_label = "1"
        self.negative_label = "0"
        
    def _load_config(self, config_file: str) -> Dict[str, str]:
        """Load configuration from JSON file."""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file '{config_file}' not found. Please create it or use the default 'sample_config.json'.")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file '{config_file}': {e}")
        
        # Validate required fields
        required_fields = ['model_name', 'target_file', 'search_criteria']
        missing_fields = [field for field in required_fields if field not in config]
        
        if missing_fields:
            raise ValueError(f"Configuration file '{config_file}' is missing required fields: {', '.join(missing_fields)}")
        
        # Validate that fields are not empty
        empty_fields = [field for field in required_fields if not config[field].strip()]
        if empty_fields:
            raise ValueError(f"Configuration file '{config_file}' has empty values for: {', '.join(empty_fields)}")
        
        return config
        
    def load_progress(self) -> Dict[str, int]:
        """Load progress from JSON file."""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {"last_paragraph": 0, "total_processed": 0, "total_found": 0}
    
    def save_progress(self, progress: Dict[str, int]):
        """Save progress to JSON file."""
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
    
    def _create_output_file_header(self):
        """Create output files with headers if they don't exist."""
        # Always create raw quotes file header
        if not self.raw_quotes_file.exists():
            with open(self.raw_quotes_file, 'w', encoding='utf-8') as f:
                pass  # Start with empty file for quotes
        
        # Create detailed results file header only if not in simple mode
        if not self.simple and not self.detailed_results_file.exists():
            with open(self.detailed_results_file, 'w', encoding='utf-8') as f:
                f.write("Social Comparison Paragraphs Found\n")
                f.write("=" * 50 + "\n\n")
    
    def append_result(self, paragraph_num: int, text: str, confidence: float, explanation: str = ""):
        """Append a result immediately to the output files."""
        # Create headers if this is the first result
        if self.results_count == 0:
            self._create_output_file_header()
        
        self.results_count += 1
        
        # Always write to raw quotes file
        with open(self.raw_quotes_file, 'a', encoding='utf-8') as f:
            f.write(f'"{text}"\n')
        
        # Write to detailed results file only if not in simple mode
        if not self.simple and explanation:
            with open(self.detailed_results_file, 'a', encoding='utf-8') as f:
                f.write(f"Finding #{self.results_count} (Paragraph {paragraph_num}, Confidence: {confidence:.2f})\n")
                f.write("-" * 60 + "\n")
                f.write(f"Explanation: {explanation}\n\n")
                f.write(f"{text}\n\n")
        
        # Only print detailed info in verbose mode
        if self.verbose:
            if self.simple:
                print(f"✓ SOCIAL COMPARISON FOUND (confidence: {confidence:.2f})")
            else:
                print(f"✓ SOCIAL COMPARISON FOUND (confidence: {confidence:.2f}) - {explanation}")
        # In non-verbose mode, the progress bar will handle the display

    def _extract_text_from_html(self, html_path: str) -> str:
        """Extract clean text from HTML file."""
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Focus on the main content - look for paragraphs
        paragraphs = soup.find_all('p')
        if paragraphs:
            # If we have proper paragraph tags, use them
            text_parts = []
            for p in paragraphs:
                p_text = p.get_text().strip()
                if len(p_text) > 20:  # Skip very short paragraphs
                    text_parts.append(p_text)
            text = '\n\n'.join(text_parts)
        else:
            # Fallback to full text extraction
            text = soup.get_text()
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text

    def _clean_paragraph(self, text: str) -> str:
        """Clean and normalize paragraph text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        # Remove page numbers and headers/footers (simple heuristic)
        text = re.sub(r'^\d+\s*$', '', text)
        return text

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        # First try splitting by double newlines
        paragraphs = re.split(r'\n\s*\n+', text)
        
        # If that doesn't give us many paragraphs, try single newlines
        if len(paragraphs) < 10:
            paragraphs = text.split('\n')
        
        # Clean and filter paragraphs
        cleaned_paragraphs = []
        for para in paragraphs:
            cleaned = self._clean_paragraph(para)
            # Skip very short paragraphs (likely page numbers, headers, etc.)
            if len(cleaned) > 30:  # Reduced minimum paragraph length
                cleaned_paragraphs.append(cleaned)
        
        return cleaned_paragraphs

    def _extract_text_from_txt(self, file_path: str) -> str:
        """Extract text from a TXT file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _split_txt_into_stanzas(self, text: str) -> List[str]:
        """Split TXT content into stanzas (groups of lines separated by blank lines)."""
        # Split by double newlines to get stanzas
        potential_stanzas = re.split(r'\n\s*\n', text)
        
        stanzas = []
        for stanza in potential_stanzas:
            # Clean up the stanza
            cleaned = stanza.strip()
            
            # Skip very short stanzas (likely headers or artifacts)
            if len(cleaned) < 50:
                continue
                
            # Skip if it looks like a title or header (all caps, short lines)
            lines = cleaned.split('\n')
            if len(lines) <= 2 and any(line.isupper() for line in lines if len(line.strip()) > 0):
                continue
                
            stanzas.append(cleaned)
        
        return stanzas

    def classify_text(self, text: str) -> tuple[str, float, str]:
        """Classify text and return label with confidence and explanation."""
        # Build full prompt from the classification description
        combined_prompt = f"""Does this text match the following criteria:
```
{self.search_criteria}
```
Please answer with "1" if it matches, "0" if it does not. You must answer with one of the two.
If you answer "1", provide a brief 1-sentence explanation of what comparison you found.
If you answer "0", just respond with "0".

Format: Either "0" or "1: [explanation]"

Text: "{text}" """
        
        # Try up to 3 times with same token limit
        max_attempts = 3
        
        for attempt in range(1, max_attempts + 1):
            if self.verbose:
                print(f"Attempt {attempt}/{max_attempts}...")
            
            response = ollama.chat(
                model=self.model_name,
                messages=[{
                    'role': 'user',
                    'content': combined_prompt
                }],
                options={
                    'temperature': 0.1,
                    'top_p': 0.9,
                    'num_predict': 4096,
                    'num_ctx': 4096,
                    'keep_alive': '10m'
                },
                stream=False
            )
            
            generated_text = response['message']['content'].strip()
            if self.verbose:
                print(f"Full model response: '{generated_text}'")
            
            # Handle DeepSeek-R1 thinking format - extract final answer after </think>
            if '</think>' in generated_text:
                # Complete thinking response
                final_answer = generated_text.split('</think>')[-1].strip()
            elif '<think>' in generated_text:
                # Truncated thinking response - retry if we have attempts left
                if attempt < max_attempts:
                    if self.verbose:
                        print(f"Response truncated (has <think> but no </think>), retrying...")
                    continue
                else:
                    if self.verbose:
                        print(f"Response still truncated after {max_attempts} attempts")
                    final_answer = ""
            else:
                # No thinking format
                final_answer = generated_text
            
            if self.verbose:
                print(f"Extracted final answer: '{final_answer}'")
            
            # Parse the response for clear answers
            if final_answer.startswith('1:'):
                explanation = final_answer[2:].strip()
                return self.positive_label, 0.95, explanation
            elif final_answer == self.positive_label:
                return self.positive_label, 0.95, "Social comparison detected."
            elif final_answer == self.negative_label:
                return self.negative_label, 0.95, ""
            elif self.positive_label in final_answer and self.negative_label not in final_answer:
                return self.positive_label, 0.85, "Social comparison detected."
            elif self.negative_label in final_answer and self.positive_label not in final_answer:
                return self.negative_label, 0.85, ""
            
            # If we get here and have attempts left, something went wrong but let's try once more
            if attempt < max_attempts:
                if self.verbose:
                    print(f"Unclear response, retrying...")
                continue
            
            # Final attempt failed - use probability resolution
            if self.verbose:
                print(f"Unclear response after {max_attempts} attempts: '{final_answer}' - checking probabilities...")
            return self._resolve_by_probability(combined_prompt)
        
        # Should never reach here
        raise RuntimeError("Failed to get clear response after all attempts")
    
    def _resolve_by_probability(self, prompt: str) -> tuple[str, float, str]:
        """When response is unclear, compare probabilities of each label."""
        prob_positive = self._get_label_probability(prompt, self.positive_label)
        prob_negative = self._get_label_probability(prompt, self.negative_label)
        
        total_prob = prob_positive + prob_negative
        if total_prob == 0:
            raise RuntimeError("Could not determine probabilities for either label")
        
        # Normalize and choose the more likely option
        norm_prob_positive = prob_positive / total_prob
        norm_prob_negative = prob_negative / total_prob
        
        if norm_prob_positive > norm_prob_negative:
            return self.positive_label, norm_prob_positive, "Social comparison detected."
        else:
            return self.negative_label, norm_prob_negative, ""
    
    def _get_label_probability(self, prompt: str, label: str) -> float:
        """Get probability of a specific label by sampling multiple times."""
        matches = 0
        total_samples = 5
        
        for _ in range(total_samples):
            response = ollama.chat(
                model=self.model_name,
                messages=[{
                    'role': 'user',
                    'content': prompt
                }],
                options={
                    'temperature': 0.3,  # Some randomness for sampling
                    'top_p': 0.9,
                    'num_predict': 4096,  # Allow reasoning like deepseek-r1
                    'keep_alive': '10m'
                }
            )
            
            generated = response['message']['content'].strip()
            
            # Handle thinking format - extract final answer
            if '</think>' in generated:
                final_answer = generated.split('</think>')[-1].strip()
            else:
                final_answer = generated
            
            # Check if the label appears in the final answer
            if label in final_answer:
                matches += 1
        
        return matches / total_samples

    def process_text_file(self, text_path: Union[str, None] = None) -> Dict[str, int]:
        """
        Process the text file and return results.
        
        Args:
            text_path: Path to the text file (if None, uses target_file from config)
            
        Returns:
            Dictionary with processing statistics
        """
        if text_path is None:
            text_path = self.config['target_file']
            
        text_file = Path(text_path)
        if not text_file.exists():
            raise FileNotFoundError(f"Text file not found: {text_file}")

        print(f"Processing text file: {text_file}")
        
        # Load progress
        progress = self.load_progress()
        start_paragraph = progress["last_paragraph"]
        total_processed = progress["total_processed"]
        total_found = progress["total_found"]
        
        if start_paragraph > 0:
            print(f"Resuming from paragraph {start_paragraph + 1}")
            if total_found > 0:
                print(f"Previously found {total_found} social comparisons")
        else:
            print(f"Analyzing from beginning")

        # Always create output file header
        self._create_output_file_header()
        
        try:
            # Determine file type and extract content accordingly
            file_extension = text_file.suffix.lower()
            
            if file_extension == '.html':
                full_text = self._extract_text_from_html(str(text_file))
                paragraphs = self._split_into_paragraphs(full_text)
                content_type = "paragraphs"
            elif file_extension == '.txt':
                full_text = self._extract_text_from_txt(str(text_file))
                paragraphs = self._split_txt_into_stanzas(full_text)
                content_type = "stanzas"
            else:
                raise ValueError(f"Unsupported file type: {file_extension}. Supported types: .html, .txt")
            
            total_paragraphs = len(paragraphs)
            
            # Prepare paragraph data for batch processing
            remaining_paragraphs = [(i, paragraph) for i, paragraph in enumerate(paragraphs[start_paragraph:], start_paragraph)]
            batch_size = 4  # Process 4 paragraphs in parallel
            
            if self.verbose:
                print(f"Total {content_type} to process: {total_paragraphs}")
                print(f"Starting from {content_type[:-1]}: {start_paragraph + 1}")
                print(f"Using batch processing with {batch_size} parallel workers")
            else:
                print(f"Processing {total_paragraphs} {content_type}...")
            
            # Start timing for ETA calculation
            import time
            self.start_time = time.time()
            
            # Process in batches
            for batch_start in range(0, len(remaining_paragraphs), batch_size):
                batch_end = min(batch_start + batch_size, len(remaining_paragraphs))
                batch_data = remaining_paragraphs[batch_start:batch_end]
                
                if self.verbose:
                    print(f"\nProcessing batch {batch_start//batch_size + 1}/{(len(remaining_paragraphs) + batch_size - 1)//batch_size}")
                
                try:
                    # Process batch in parallel
                    batch_results = self._process_paragraph_batch(batch_data, batch_size)
                    
                    # Handle results
                    for paragraph_num, response, confidence, explanation in batch_results:
                        total_processed += 1
                        
                        if response == '1':
                            total_found += 1
                            self.append_result(paragraph_num + 1, paragraphs[paragraph_num], confidence, explanation)
                            result_status = f"● found (conf: {confidence:.2f})"
                        else:
                            result_status = f"○ skip (conf: {confidence:.2f})"
                        
                        if self.verbose:
                            print(f"  {content_type[:-1].capitalize()} {paragraph_num}/{total_paragraphs}: {result_status}")
                        else:
                            # Update progress bar with total found count
                            self._update_progress(total_processed, total_paragraphs, total_found)
                        
                        # Save progress after each paragraph
                        progress = {
                            "last_paragraph": paragraph_num,
                            "total_processed": total_processed,
                            "total_found": total_found
                        }
                        self.save_progress(progress)
                        
                except KeyboardInterrupt:
                    print(f"\n\nProcessing interrupted by user.")
                    print(f"Progress saved. Resume by running the script again.")
                    return {
                        "last_paragraph": total_processed + start_paragraph,
                        "total_processed": total_processed,
                        "total_found": total_found
                    }
                
        except KeyboardInterrupt:
            print(f"\n\nProcessing interrupted by user.")
            print(f"Progress saved. Resume by running the script again.")
            return progress
        except Exception as e:
            print(f"Error processing text: {e}")
            raise
        
        print(f"\nProcessing complete!")
        print(f"Total paragraphs processed: {total_processed}")
        print(f"Social comparisons found: {total_found}")
        if total_processed > 0:
            print(f"Success rate: {total_found/total_processed*100:.1f}%")
        
        # Add completion summary to output file
        with open(self.detailed_results_file, 'a', encoding='utf-8') as f:
            f.write("\n" + "=" * 50 + "\n")
            f.write("PROCESSING COMPLETE\n")
            f.write("=" * 50 + "\n")
            f.write(f"Total paragraphs processed: {total_processed}\n")
            f.write(f"Social comparisons found: {total_found}\n")
            if total_processed > 0:
                f.write(f"Success rate: {total_found/total_processed*100:.1f}%\n")
        
        # Clear progress file on completion
        self.progress_file.unlink(missing_ok=True)
        
        return progress

    def _update_progress(self, current: int, total: int, total_found: int = 0):
        """Update progress bar in place."""
        if not self.verbose:
            # Create a simple progress bar
            progress = current / total
            bar_length = 50
            filled_length = int(bar_length * progress)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            
            # Calculate ETA if we have timing data
            status = f"[{current:4d}/{total}] {bar} {progress:.1%}"
            
            if total_found > 0:
                status += f" | Found: {total_found}"
            
            # Add ETA if we have enough data
            if self.start_time and current > 0:
                import time
                elapsed = time.time() - self.start_time
                avg_time_per_item = elapsed / current
                remaining_items = total - current
                eta_seconds = remaining_items * avg_time_per_item
                
                if eta_seconds > 60:
                    eta_minutes = int(eta_seconds / 60)
                    eta_seconds = int(eta_seconds % 60)
                    status += f" | ETA: {eta_minutes}m{eta_seconds:02d}s"
                else:
                    status += f" | ETA: {int(eta_seconds)}s"
            
            # Use \r to overwrite the current line
            print(f"\r{status}", end="", flush=True)
            
            # Print newline only when complete
            if current == total:
                print()  # Final newline

    def _get_run_directory(self) -> Path:
        """Get the appropriate run directory (numbered subdirectory)."""
        existing_runs = [
            int(d.name) for d in self.base_output_dir.iterdir() 
            if d.is_dir() and d.name.isdigit()
        ]
        
        if self.directory is not None:
            # Use specific directory number
            run_dir = self.base_output_dir / str(self.directory)
            if self.verbose:
                action = "clearing and using" if self.restart else "using"
                print(f"Directory specified: {action} {run_dir}")
        else:
            # Default: create next available directory
            next_run = max(existing_runs, default=-1) + 1
            run_dir = self.base_output_dir / str(next_run)
            if self.verbose:
                action = "clearing and creating" if self.restart else "creating"
                print(f"No directory specified: {action} new directory {run_dir}")
        
        # Clear files if restart flag is set
        if self.restart:
            if run_dir.exists():
                progress_file = run_dir / "progress.json"
                raw_quotes_file = run_dir / "raw_quotes.txt"
                detailed_results_file = run_dir / "detailed_results.txt"
                
                for file_to_clear in [progress_file, raw_quotes_file, detailed_results_file]:
                    if file_to_clear.exists():
                        file_to_clear.unlink()
                        if self.verbose:
                            print(f"Cleared existing file: {file_to_clear}")
            elif self.verbose:
                print(f"No existing files to clear in {run_dir}")
        
        return run_dir

    def _process_paragraph_batch(self, paragraph_data: List[Tuple[int, str]], batch_size: int = 4) -> List[Tuple[int, str, float, str]]:
        """Process a batch of paragraphs in parallel."""
        results = []
        
        def process_single_paragraph(paragraph_info):
            paragraph_num, paragraph = paragraph_info
            try:
                response, confidence, explanation = self.classify_text(paragraph)
                return (paragraph_num, response, confidence, explanation)
            except Exception as e:
                if self.verbose:
                    print(f"\nError processing paragraph {paragraph_num}: {e}")
                return (paragraph_num, '0', 0.0, "")
        
        # Process in batches to avoid overwhelming the model
        for i in range(0, len(paragraph_data), batch_size):
            batch = paragraph_data[i:i + batch_size]
            
            with ThreadPoolExecutor(max_workers=min(batch_size, 4)) as executor:
                # Submit all tasks in the batch
                future_to_paragraph = {
                    executor.submit(process_single_paragraph, para_info): para_info
                    for para_info in batch
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_paragraph):
                    paragraph_info = future_to_paragraph[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        paragraph_num = paragraph_info[0]
                        if self.verbose:
                            print(f"\nError in batch processing for paragraph {paragraph_num}: {e}")
                        results.append((paragraph_num, '0', 0.0, ""))
        
        # Sort results by paragraph number to maintain order
        results.sort(key=lambda x: x[0])
        return results

def verify_model_available(model_name: str):
    """Verify that the specified model is available in Ollama."""
    models_response = ollama.list()
    
    # Extract model names from the response
    model_names = [model.model for model in models_response.models]
    
    print(f"Available models: {model_names}")
    
    if model_name not in model_names:
        raise ValueError(f"Model '{model_name}' not found. Available models: {', '.join(model_names)}. Run: ollama pull {model_name}")
    
    print(f"Using model: {model_name}")


def is_ollama_running() -> bool:
    """Check if Ollama is currently running by trying to connect."""
    try:
        ollama.list()
        return True
    except Exception:
        return False

def start_ollama(verbose: bool = False) -> bool:
    """Start Ollama if it's not running. Returns True if successful."""
    if is_ollama_running():
        if verbose:
            print("Ollama is already running.")
        return True
    
    if verbose:
        print("Ollama is not running. Attempting to start it...")
    
    try:
        # Try to start Ollama as a background process
        if sys.platform.startswith('win'):
            # Windows
            subprocess.Popen(['ollama', 'serve'], creationflags=0x00000010)  # CREATE_NEW_CONSOLE
        else:
            # Unix-like systems (Mac, Linux)
            subprocess.Popen(['ollama', 'serve'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Wait a bit for Ollama to start
        if verbose:
            print("Waiting for Ollama to start...")
        
        for i in range(10):  # Wait up to 10 seconds
            time.sleep(1)
            if is_ollama_running():
                if verbose:
                    print("Ollama started successfully!")
                return True
            if verbose:
                print(f"Still waiting... ({i+1}/10)")
        
        if verbose:
            print("Timeout waiting for Ollama to start.")
        return False
        
    except FileNotFoundError:
        if verbose:
            print("Error: 'ollama' command not found. Please install Ollama first.")
        return False
    except Exception as e:
        if verbose:
            print(f"Error starting Ollama: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Extract social comparison paragraphs from text using a local LLM")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed debug output")
    parser.add_argument("--config", "-c", default="sample_config.json", help="Configuration file path")
    parser.add_argument("--simple", "-s", action="store_true", help="Simple mode: only generate raw quotes without explanations")
    parser.add_argument("--restart", "-r", action="store_true", help="Clear files in the specified directory (use with -d)")
    parser.add_argument("--directory", "-d", type=int, help="Use specific directory number (e.g., -d 5 for output/5/)")
    args = parser.parse_args()
    
    print("Text Classifier")
    print("=" * 30)
    
    # Start Ollama if it's not running
    if not start_ollama(verbose=args.verbose):
        print("Error: Could not start Ollama. Please make sure Ollama is installed and try running 'ollama serve' manually.")
        sys.exit(1)
    
    # Initialize classifier with config file
    try:
        classifier = TextClassifier(
            config_file=args.config, 
            verbose=args.verbose, 
            simple=args.simple, 
            restart=args.restart,
            directory=args.directory
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"Configuration Error: {e}")
        sys.exit(1)
    
    # Check if model is available
    verify_model_available(classifier.model_name)
    
    # Test the model connection
    if args.verbose:
        print("Testing Ollama connection...")
    try:
        ollama.chat(
            model=classifier.model_name, 
            messages=[{'role': 'user', 'content': 'test'}], 
            options={'num_predict': 1}
        )
        if args.verbose:
            print("Ollama connection successful!")
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        print("Please check that Ollama is running and the model is available.")
        sys.exit(1)
    
    # Process the text file
    classifier.process_text_file()
    
    print(f"\nResults saved to: {classifier.output_dir}")
    if args.simple:
        print(f"- Raw quotes: {classifier.raw_quotes_file}")
    else:
        print(f"- Raw quotes: {classifier.raw_quotes_file}")
        print(f"- Detailed results: {classifier.detailed_results_file}")
    if args.verbose:
        print(f"Progress file: {classifier.progress_file}")


if __name__ == "__main__":
    main()
