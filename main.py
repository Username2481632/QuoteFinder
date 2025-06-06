#!/usr/bin/env python3
"""
Quote Finder - Extract matching paragraphs from text using a local LLM

This script processes HTML/text files paragraph by paragraph, sends each paragraph
to a local language model via Ollama, and logs paragraphs that match
specified criteria.

Requirements:
- Ollama installed and running
- DeepSeek-R1 distill model pulled in Ollama
- Python packages: beautifulsoup4, ollama

Usage:
    python main.py [--verbose]
"""

import argparse
import json
import os
import re
import signal
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
        
        # Support model_names configuration
        if 'model_names' not in self.config:
            raise ValueError("Configuration must contain 'model_names' (list of model names)")
            
        self.model_names = self.config['model_names']
        if not isinstance(self.model_names, list):
            raise ValueError("model_names must be a list of model names")
        if len(self.model_names) == 0:
            raise ValueError("model_names cannot be empty")
            
        self.verbose = verbose
        self.simple = simple
        self.restart = restart
        self.directory = directory
        self.prompt_template = self.config['search_criteria']
        self.search_criteria = self.config['search_criteria']
        
        # Progress tracking
        self.start_time = None
        self.session_start_processed = 0  # How many were processed when this session started
        self.cancelled = False  # Flag for cancellation
        
        # Current progress state (updated during processing)
        self.current_paragraph = 0
        self.current_total_processed = 0
        self.current_total_found = 0
        
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
        required_fields = ['model_names', 'target_file', 'search_criteria']
        missing_fields = [field for field in required_fields if field not in config]
        
        if missing_fields:
            raise ValueError(f"Configuration file '{config_file}' is missing required fields: {', '.join(missing_fields)}")
        
        # Validate that fields are not empty
        empty_fields = []
        for field in required_fields:
            value = config[field]
            if field == 'model_names':
                # model_names should be a non-empty list
                if not isinstance(value, list) or len(value) == 0:
                    empty_fields.append(field)
            else:
                # Other fields should be non-empty strings
                if not value or not value.strip():
                    empty_fields.append(field)
        
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
                f.write("Matching Paragraphs Found\n")
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
                print(f"✓ MATCH FOUND (confidence: {confidence:.2f})")
            else:
                print(f"✓ MATCH FOUND (confidence: {confidence:.2f}) - {explanation}")
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

    def classify_text(self, text: str, model_name: str, need_explanation: bool = True) -> tuple[str, float, str]:
        """Classify text and return label with confidence and explanation."""
        # Build prompt - simpler for cascade models that don't need explanations
        if need_explanation:
            combined_prompt = f"""Does this text match the following criteria:
```
{self.search_criteria}
```
Please answer with "1" if it matches, "0" if it does not. You must answer with one of the two.
If you answer "1", provide a brief 1-sentence explanation of what you found.
If you answer "0", just respond with "0".

Format: Either "0" or "1: [explanation]"

Text: "{text}" """
        else:
            combined_prompt = f"""Does this text match the following criteria:
```
{self.search_criteria}
```
Please answer with "1" if it matches, "0" if it does not. You must answer with one of the two.

Text: "{text}" """
        
        # Try up to 3 times with same token limit
        max_attempts = 3
        
        for attempt in range(1, max_attempts + 1):
            response = ollama.chat(
                model=model_name,
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
            
            # Handle DeepSeek-R1 thinking format - extract final answer after </think>
            if '</think>' in generated_text:
                # Complete thinking response
                final_answer = generated_text.split('</think>')[-1].strip()
            elif '<think>' in generated_text:
                # Truncated thinking response - retry if we have attempts left
                if attempt < max_attempts:
                    continue
                else:
                    final_answer = ""
            else:
                # No thinking format
                final_answer = generated_text
            
            # Parse the response for clear answers
            if final_answer.startswith('1:'):
                explanation = final_answer[2:].strip()
                return self.positive_label, 0.95, explanation
            elif final_answer == self.positive_label:
                return self.positive_label, 0.95, "Match detected."
            elif final_answer == self.negative_label:
                return self.negative_label, 0.95, ""
            elif self.positive_label in final_answer and self.negative_label not in final_answer:
                return self.positive_label, 0.85, "Match detected."
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
            return self._resolve_by_probability(combined_prompt, model_name)
        
        # Should never reach here
        raise RuntimeError("Failed to get clear response after all attempts")
    
    def _resolve_by_probability(self, prompt: str, model_name: str) -> tuple[str, float, str]:
        """When response is unclear, compare probabilities of each label."""
        prob_positive = self._get_label_probability(prompt, self.positive_label, model_name)
        prob_negative = self._get_label_probability(prompt, self.negative_label, model_name)
        
        total_prob = prob_positive + prob_negative
        if total_prob == 0:
            raise RuntimeError("Could not determine probabilities for either label")
        
        # Normalize and choose the more likely option
        norm_prob_positive = prob_positive / total_prob
        norm_prob_negative = prob_negative / total_prob
        
        if norm_prob_positive > norm_prob_negative:
            return self.positive_label, norm_prob_positive, "Match detected."
        else:
            return self.negative_label, norm_prob_negative, ""
    
    def _get_label_probability(self, prompt: str, label: str, model_name: str) -> float:
        """Get probability of a specific label by sampling multiple times."""
        matches = 0
        total_samples = 5
        
        for _ in range(total_samples):
            response = ollama.chat(
                model=model_name,
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

    def classify_with_cascade(self, text: str, progress_callback=None, stanza_info="") -> tuple[str, float, str, str]:
        """
        Classify text using cascade verification with multiple models.
        First model detects positives, subsequent models verify them.
        Returns final classification after all models agree.
        Returns: (response, confidence, explanation, cascade_info)
        
        Args:
            text: Text to classify
            progress_callback: Optional callback for real-time progress updates
            stanza_info: Info string for progress display (e.g., "Stanza 75/634")
        """
        if len(self.model_names) == 1:
            # Single model - use standard classification
            response, confidence, explanation = self.classify_text(text, self.model_names[0])
            cascade_info = f"Model 1: {response}"
            if progress_callback:
                progress_callback(stanza_info, cascade_info, response == '1')
            return response, confidence, explanation, cascade_info
        
        # Multi-model cascade verification
        explanation = ""
        cascade_decisions = []
        
        for i, model_name in enumerate(self.model_names):
            # Only final model needs to generate explanations
            is_final_model = (i == len(self.model_names) - 1)
            response, confidence, explanation = self.classify_text(text, model_name, need_explanation=is_final_model)
            cascade_decisions.append(f"Model {i+1}: {response}")
            
            # Send progress update after each model
            current_cascade = " | ".join(cascade_decisions)
            if progress_callback:
                progress_callback(stanza_info, current_cascade, None)  # None = still in progress
            
            if response == self.negative_label:
                # If any model in the cascade says negative, reject
                cascade_decisions[-1] += " → rejected"
                cascade_info = " | ".join(cascade_decisions)
                if progress_callback:
                    progress_callback(stanza_info, cascade_info, False, final=True)
                return self.negative_label, confidence, "", cascade_info
        
        # All models agreed it's positive - use final model's explanation
        cascade_decisions.append(f"→ All {len(self.model_names)} approved")
        cascade_info = " | ".join(cascade_decisions)
        if progress_callback:
            progress_callback(stanza_info, cascade_info, True, final=True)
        
        return self.positive_label, 0.95, explanation, cascade_info

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
        
        # Initialize current progress state
        self.current_paragraph = start_paragraph
        self.current_total_processed = total_processed
        self.current_total_found = total_found
        
        if start_paragraph > 0:
            print(f"Resuming from paragraph {start_paragraph + 1}")
            if total_found > 0:
                print(f"Previously found {total_found} matches")
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
            self.session_start_processed = total_processed  # Track work done before this session
            
            # Process in batches
            for batch_start in range(0, len(remaining_paragraphs), batch_size):
                # Check for cancellation
                if self.cancelled:
                    break
                    
                batch_end = min(batch_start + batch_size, len(remaining_paragraphs))
                batch_data = remaining_paragraphs[batch_start:batch_end]
                
                if self.verbose:
                    batch_num = batch_start//batch_size + 1
                    total_batches = (len(remaining_paragraphs) + batch_size - 1)//batch_size
                    print(f"\nBatch {batch_num}/{total_batches}")
                
                try:
                    # Process batch with real-time cascade visualization
                    self._process_batch_with_live_cascade(batch_data, paragraphs, content_type, total_paragraphs)
                    
                    # Update totals based on what was actually processed
                    total_processed = self.current_total_processed
                    total_found = self.current_total_found
                    
                    # Check if cancelled after processing batch results
                    if self.cancelled:
                        break
                        
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
        
        # Check if processing was cancelled
        if self.cancelled:
            print(f"\n\nProcessing cancelled by user.")
            print(f"Progress saved. Resume by running the script again.")
            return {
                "last_paragraph": total_processed + start_paragraph,
                "total_processed": total_processed,
                "total_found": total_found
            }
        
        print(f"\nProcessing complete!")
        print(f"Total paragraphs processed: {total_processed}")
        print(f"Matches found: {total_found}")
        if total_processed > 0:
            print(f"Success rate: {total_found/total_processed*100:.1f}%")
        
        # Add completion summary to output file
        with open(self.detailed_results_file, 'a', encoding='utf-8') as f:
            f.write("\n" + "=" * 50 + "\n")
            f.write("PROCESSING COMPLETE\n")
            f.write("=" * 50 + "\n")
            f.write(f"Total paragraphs processed: {total_processed}\n")
            f.write(f"Matches found: {total_found}\n")
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
            if self.start_time and current > self.session_start_processed:
                import time
                elapsed = time.time() - self.start_time
                session_work_done = current - self.session_start_processed
                avg_time_per_item = elapsed / session_work_done
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
                response, confidence, explanation, cascade_info = self.classify_with_cascade(paragraph)
                return (paragraph_num, response, confidence, explanation, cascade_info)
            except Exception as e:
                if self.verbose:
                    print(f"\nError processing paragraph {paragraph_num}: {e}")
                return (paragraph_num, '0', 0.0, "", "Error")
        
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
                        
                        # Update progress immediately for signal handler access
                        paragraph_num, response, confidence, explanation, cascade_info = result
                        self.current_paragraph = paragraph_num
                        self.current_total_processed += 1
                        if response == '1':
                            self.current_total_found += 1
                            
                    except Exception as e:
                        paragraph_num = paragraph_info[0]
                        if self.verbose:
                            print(f"\nError in batch processing for paragraph {paragraph_num}: {e}")
                        results.append((paragraph_num, '0', 0.0, "", "Error"))
                        
                        # Update progress even for errors
                        self.current_paragraph = paragraph_num
                        self.current_total_processed += 1
        
        # Sort results by paragraph number to maintain order
        results.sort(key=lambda x: x[0])
        return results

    def _process_batch_with_live_cascade(self, batch_data, paragraphs, content_type, total_paragraphs):
        """Process batch with live cascade visualization - each model call is separate."""
        import sys
        
        # Track state for each stanza and current display line
        stanza_states = {}
        completed_stanzas = set()
        last_displayed_stanza = None
        
        # Initialize display state
        for i, (paragraph_num, paragraph) in enumerate(batch_data):
            stanza_states[paragraph_num] = {
                'paragraph': paragraph,
                'model_idx': 0,
                'decisions': [],
                'final_result': None,
                'has_printed_line': False
            }
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            # Submit initial Model 1 tasks for all stanzas
            active_futures = {}
            
            for paragraph_num, paragraph in batch_data:
                if len(self.model_names) > 0:
                    future = executor.submit(self.classify_text, paragraph, self.model_names[0], need_explanation=False)
                    active_futures[future] = (paragraph_num, 0)
            
            # Process results as they complete
            while active_futures and not self.cancelled:
                for future in as_completed(active_futures):
                    paragraph_num, model_idx = active_futures[future]
                    del active_futures[future]
                    
                    if paragraph_num in completed_stanzas:
                        continue
                    
                    try:
                        response, confidence, explanation = future.result()
                        state = stanza_states[paragraph_num]
                        state['decisions'].append(f"Model {model_idx + 1}: {response}")
                        
                        if response == self.negative_label:
                            # Rejected - finalize this stanza
                            state['decisions'][-1] += " → rejected"
                            state['final_result'] = (response, confidence, "")
                            self._display_final_result(state, paragraph_num, total_paragraphs, content_type, paragraphs)
                            completed_stanzas.add(paragraph_num)
                            
                        elif model_idx + 1 < len(self.model_names):
                            # Continue to next model - show progress with overwriting
                            decisions_str = " | ".join(state['decisions'])
                            line_text = f"  {content_type[:-1].capitalize()} {paragraph_num}/{total_paragraphs}: {decisions_str}"
                            
                            if state['has_printed_line']:
                                # Overwrite the existing line for this stanza
                                print(f"\r{line_text}", end='', flush=True)
                            else:
                                # First time showing this stanza
                                print(line_text)
                                state['has_printed_line'] = True
                            
                            # Submit next model
                            is_final_model = (model_idx + 1 == len(self.model_names) - 1)
                            next_future = executor.submit(
                                self.classify_text, 
                                state['paragraph'], 
                                self.model_names[model_idx + 1], 
                                need_explanation=is_final_model
                            )
                            active_futures[next_future] = (paragraph_num, model_idx + 1)
                            state['model_idx'] = model_idx + 1
                            
                        else:
                            # All models approved - finalize
                            state['decisions'].append(f"→ All {len(self.model_names)} approved")
                            state['final_result'] = (response, 0.95, explanation)
                            
                            # If we had been showing intermediate progress, clear the line first
                            if state['has_printed_line']:
                                print()  # Move to new line before final result
                            
                            self._display_final_result(state, paragraph_num, total_paragraphs, content_type, paragraphs)
                            completed_stanzas.add(paragraph_num)
                        
                    except Exception as e:
                        if self.verbose:
                            print(f"\nError processing stanza {paragraph_num}, model {model_idx + 1}: {e}")
                        state = stanza_states[paragraph_num]
                        state['final_result'] = ('0', 0.0, "")
                        
                        # If we had been showing intermediate progress, clear the line first
                        if state['has_printed_line']:
                            print()  # Move to new line before final result
                            
                        self._display_final_result(state, paragraph_num, total_paragraphs, content_type, paragraphs)
                        completed_stanzas.add(paragraph_num)
                    
                    # Break inner loop to restart as_completed with updated active_futures
                    break
    
    def _display_final_result(self, state, paragraph_num, total_paragraphs, content_type, paragraphs):
        """Display the final result for a stanza."""
        response, confidence, explanation = state['final_result']
        
        # Update progress tracking
        self.current_paragraph = paragraph_num
        self.current_total_processed += 1
        
        if response == '1':
            self.current_total_found += 1
            self.append_result(paragraph_num + 1, paragraphs[paragraph_num], confidence, explanation)
            result_status = f"● found (conf: {confidence:.2f})"
        else:
            result_status = f"○ skip (conf: {confidence:.2f})"
        
        # Final display - always on a clean new line
        decisions_str = " | ".join(state['decisions'])
        print(f"  {content_type[:-1].capitalize()} {paragraph_num}/{total_paragraphs}: {decisions_str} → {result_status}")
        
        # Save progress
        progress = {
            "last_paragraph": paragraph_num,
            "total_processed": self.current_total_processed,
            "total_found": self.current_total_found
        }
        self.save_progress(progress)

def verify_models_available(model_names: Union[str, List[str]]):
    """Verify that the specified model(s) are available in Ollama."""
    models_response = ollama.list()
    
    # Extract model names from the response
    available_models = [model.model for model in models_response.models]
    
    print(f"Available models: {available_models}")
    
    # Support both single model (backward compatibility) and multiple models
    if isinstance(model_names, str):
        models_to_check = [model_names]
    else:
        models_to_check = model_names
    
    missing_models = []
    for model_name in models_to_check:
        if model_name not in available_models:
            missing_models.append(model_name)
    
    if missing_models:
        missing_str = ', '.join(missing_models)
        available_str = ', '.join(available_models)
        pull_commands = ' && '.join([f"ollama pull {model}" for model in missing_models])
        raise ValueError(f"Model(s) {missing_str} not found. Available models: {available_str}. Run: {pull_commands}")
    
    if len(models_to_check) == 1:
        print(f"Using model: {models_to_check[0]}")
    else:
        print(f"Using cascade verification with models: {' → '.join(models_to_check)}")


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
    parser = argparse.ArgumentParser(description="Extract matching paragraphs from text using a local LLM")
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
    
    # Check if models are available and preload them
    verify_models_available(classifier.model_names)
    
    # Set up signal handler for graceful cancellation
    global classifier_instance
    classifier_instance = classifier
    signal.signal(signal.SIGINT, signal_handler)
    
    # Test the model connection
    if args.verbose:
        print("Testing Ollama connection...")
    try:
        ollama.chat(
            model=classifier.model_names[0], 
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


# Global variable to hold classifier instance for signal handler
classifier_instance = None

def signal_handler(signum, frame):
    """Handle Ctrl+C by saving progress and terminating immediately."""
    global classifier_instance
    
    print("\n\nReceived cancellation signal (Ctrl+C). Saving progress and stopping immediately...")
    
    # Save current progress - classifier_instance is guaranteed to exist
    try:
        current_progress = {
            "last_paragraph": classifier_instance.current_paragraph,
            "total_processed": classifier_instance.current_total_processed,
            "total_found": classifier_instance.current_total_found
        }
        classifier_instance.save_progress(current_progress)
        print(f"Progress saved (processed: {current_progress['total_processed']}, found: {current_progress['total_found']}).")
    except Exception as e:
        print(f"Could not save progress: {e}")
    
    print("Cancelled.")
    
    # Flush output buffers to ensure messages are shown
    sys.stdout.flush()
    sys.stderr.flush()
    
    # Force immediate termination - bypasses cleanup and Python exit handlers
    os._exit(1)


if __name__ == "__main__":
    main()
