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
from typing import List, Tuple, Dict, Union, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
import ollama


class TextClassifier:
    def __init__(self, config_file: Optional[str] = None, verbose: bool = False, simple: bool = False, restart: bool = False, directory: Union[None, int] = None):
        """
        Initialize the Text Classifier.
        
        Args:
            config_file: Path to the configuration JSON file (None to auto-detect)
            verbose: Whether to show detailed debug output
            simple: If True, only generate raw quotes without explanations
            restart: If True, start fresh with next available directory number
            directory: If provided, use this specific directory number
        """
        # Store parameters for later use in config resolution
        self.verbose = verbose
        self.simple = simple
        self.restart = restart
        self.directory = directory
        
        # Create main output directory early
        self.base_output_dir = Path("output")
        self.base_output_dir.mkdir(exist_ok=True)
        
        # Determine the run directory first
        self.output_dir = self._get_run_directory()
        self.output_dir.mkdir(exist_ok=True)
        
        # Resolve and load config
        resolved_config_file = self._resolve_config_file(config_file)
        self.config = self._load_config(resolved_config_file)
        
        # Copy config to output directory if a config file was specified
        if config_file is not None:
            self._copy_config_to_output(config_file)
        
        # Support model_names configuration
        if 'model_names' not in self.config:
            raise ValueError("Configuration must contain 'model_names' (list of model names)")
            
        self.model_names = self.config['model_names']
        if not isinstance(self.model_names, list):
            raise ValueError("model_names must be a list of model names")
        if len(self.model_names) == 0:
            raise ValueError("model_names cannot be empty")
            
        self.prompt_template = self.config['search_criteria']
        self.search_criteria = self.config['search_criteria']
        
        # Progress tracking
        self.start_time: Optional[float] = None
        self.session_start_processed: int = 0  # How many were processed when this session started
        self.cancelled: bool = False  # Flag for cancellation
        
        # Current progress state (updated during processing)
        self.current_paragraph: int = 0
        self.current_total_processed: int = 0
        self.current_total_found: int = 0
        self.completed_paragraphs: set[int] = set()  # Track which paragraphs are completed
        
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
        required_fields: List[str] = ['model_names', 'target_file', 'search_criteria']
        missing_fields: List[str] = [field for field in required_fields if field not in config]
        
        if missing_fields:
            raise ValueError(f"Configuration file '{config_file}' is missing required fields: {', '.join(missing_fields)}")
        
        # Validate that fields are not empty
        empty_fields: List[str] = []
        for field in required_fields:
            value = config[field]
            if field == 'model_names':
                # model_names should be a non-empty list
                if not isinstance(value, list) or not value:
                    empty_fields.append(field)
            else:
                # Other fields should be non-empty strings
                if not value or not value.strip():
                    empty_fields.append(field)
        
        if empty_fields:
            raise ValueError(f"Configuration file '{config_file}' has empty values for: {', '.join(empty_fields)}")
        
        return config
    
    def _resolve_config_file(self, config_file: Optional[str]) -> str:
        """
        Resolve the configuration file path.
        If config_file is provided, use it.
        If not provided and using existing directory, look for config.json in that directory.
        Otherwise, error out.
        """
        if config_file is not None:
            # Config file explicitly provided
            return config_file
        
        # No config file provided - look for config.json in the output directory
        if self.directory is not None:
            # Using specific directory - look for config.json there
            config_path = self.output_dir / "config.json"
            if config_path.exists():
                if self.verbose:
                    print(f"Using existing config file: {config_path}")
                return str(config_path)
            else:
                raise FileNotFoundError(
                    f"No config file specified and no config.json found in directory {self.output_dir}. "
                    f"Please provide a config file with -c or ensure the directory has a config.json file."
                )
        else:
            # Creating new directory but no config provided
            raise FileNotFoundError(
                "No config file specified. Please provide a config file with -c parameter."
            )
    
    def _copy_config_to_output(self, source_config_file: str) -> None:
        """Copy the source config file to config.json in the output directory."""
        import shutil
        
        source_path = Path(source_config_file)
        dest_path = self.output_dir / "config.json"
        
        if source_path.exists():
            shutil.copy2(source_path, dest_path)
            if self.verbose:
                print(f"Copied config file from {source_path} to {dest_path}")
        else:
            # This shouldn't happen since _load_config would have failed first
            raise FileNotFoundError(f"Source config file {source_path} not found")
        
    def _compress_paragraph_list(self, paragraphs: List[int]) -> List[Union[int, List[int]]]:
        """Compress a list of paragraph numbers into ranges for efficient storage.
        
        Examples:
        [0, 1, 2, 5, 7, 8, 9] -> [range(0, 2), 5, range(7, 9)]
        [1, 3, 5, 7] -> [1, 3, 5, 7]
        [0, 1, 2, 3, 4] -> [range(0, 4)]
        """
        if not paragraphs:
            return []
        
        sorted_paragraphs = sorted(set(paragraphs))
        compressed: List[Union[int, List[int]]] = []
        start = sorted_paragraphs[0]
        end = start
        
        for i in range(1, len(sorted_paragraphs)):
            if sorted_paragraphs[i] == end + 1:
                # Continue the current range
                end = sorted_paragraphs[i]
            else:
                # End current range and start a new one
                if start == end:
                    compressed.append(start)  # Single number
                else:
                    compressed.append([start, end])  # Range as [start, end]
                start = sorted_paragraphs[i]
                end = start
        
        # Add the final range
        if start == end:
            compressed.append(start)
        else:
            compressed.append([start, end])
        
        return compressed
    
    def _decompress_paragraph_list(self, compressed: List[Union[int, List[int]]]) -> List[int]:
        """Decompress ranges back into a full list of paragraph numbers."""
        paragraphs: List[int] = []
        for item in compressed:
            if isinstance(item, int):
                paragraphs.append(item)
            elif len(item) == 2:  # It's a list with 2 elements
                start, end = item
                paragraphs.extend(range(start, end + 1))
        return sorted(paragraphs)

    def load_progress(self) -> Dict[str, Any]:
        """Load progress from JSON file with decompression."""
        try:
            if self.progress_file.exists():
                with open(self.progress_file, 'r') as f:
                    progress = json.load(f)
                    
                # Decompress the completed_paragraphs list
                if 'completed_paragraphs' in progress:
                    progress['completed_paragraphs'] = self._decompress_paragraph_list(progress['completed_paragraphs'])
                    
                return progress
        except Exception as e:
            if self.verbose:
                print(f"Error loading progress: {e}")
        
        # Return empty progress state
        return {
            'completed_paragraphs': [],
            'total_found': 0
        }

    def save_progress(self, progress: Dict[str, Any]) -> None:
        """Save progress to JSON file with compression."""
        # Compress the completed_paragraphs list before saving
        compressed_progress = progress.copy()
        if 'completed_paragraphs' in compressed_progress:
            compressed_progress['completed_paragraphs'] = self._compress_paragraph_list(progress['completed_paragraphs'])
        
        with open(self.progress_file, 'w') as f:
            json.dump(compressed_progress, f, indent=2)

    def _create_output_file_header(self) -> None:
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
    
    def append_result(self, paragraph_num: int, text: str, confidence: float, explanation: str = "") -> None:
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
            text_parts: List[str] = []
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
        cleaned_paragraphs: List[str] = []
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
        
        stanzas: List[str] = []
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

    def classify_text(self, text: str, model_name: str, is_final_model: bool = True) -> Tuple[str, float, str]:
        """Classify text and return label with confidence and explanation."""
        # Build prompt with conditional explanation instructions
        combined_prompt = f"""Does this text match the following criteria:
```
{self.search_criteria}
```
Please answer with "1" if it matches, "0" if it does not. You must answer with one of the two.
{'''If you answer "1", provide a brief 1-sentence explanation of what you found.
If you answer "0", just respond with "0".

Format: Either "0" or "1: [explanation]"''' if is_final_model else "If unsure, put 1."}

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

    def classify_with_cascade(self, text: str, progress_callback: Optional[Callable[..., None]] = None, stanza_info: str = "") -> Tuple[str, float, str, str]:
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
        explanation: str = ""
        cascade_decisions: List[str] = []
        
        for i, model_name in enumerate(self.model_names):
            # Only final model needs to generate explanations
            is_final_model = (i == len(self.model_names) - 1)
            response, confidence, explanation = self.classify_text(text, model_name, is_final_model=is_final_model)
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

    def process_text_file(self, text_path: Union[str, None] = None) -> Dict[str, Any]:
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
        completed_paragraphs = set(progress["completed_paragraphs"])
        total_found = progress["total_found"]
        total_processed = len(completed_paragraphs)  # Derived from completed list
        
        # Initialize current progress state
        self.current_total_processed = total_processed
        self.current_total_found = total_found
        self.completed_paragraphs = completed_paragraphs
        
        if completed_paragraphs:
            max_completed = max(completed_paragraphs) if completed_paragraphs else -1
            print(f"Resuming: {len(completed_paragraphs)} paragraphs already completed (up to #{max_completed + 1})")
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
            
            # Prepare paragraph data for continuous processing - skip already completed paragraphs
            remaining_paragraphs = [(i, paragraph) for i, paragraph in enumerate(paragraphs) if i not in completed_paragraphs]
            
            if self.verbose:
                print(f"Total {content_type} to process: {total_paragraphs}")
                print(f"Remaining {content_type}: {len(remaining_paragraphs)}")
                print(f"Using continuous processing with up to 8 parallel workers")
            else:
                print(f"Processing {len(remaining_paragraphs)} remaining {content_type}...")
            
            # Start timing for ETA calculation
            import time
            self.start_time = time.time()
            self.session_start_processed = total_processed  # Track work done before this session
            
            try:
                # Process all paragraphs continuously
                if self.verbose:
                    self._process_paragraphs_continuously(remaining_paragraphs, paragraphs, content_type, total_paragraphs, verbose=True)
                else:
                    self._process_paragraphs_continuously(remaining_paragraphs, paragraphs, content_type, total_paragraphs, verbose=False)
                
                # Update totals based on what was actually processed
                total_processed = self.current_total_processed
                total_found = self.current_total_found
                    
            except KeyboardInterrupt:
                print(f"\n\nProcessing interrupted by user.")
                print(f"Progress saved. Resume by running the script again.")
                return {
                    "completed_paragraphs": list(self.completed_paragraphs),
                    "total_found": self.current_total_found
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
                "completed_paragraphs": list(self.completed_paragraphs),
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
        
        return {
            "completed_paragraphs": list(self.completed_paragraphs),
            "total_found": total_found
        }

    def _format_eta(self, eta_seconds: float) -> str:
        """Format ETA seconds into a compact, readable string."""
        if eta_seconds >= 86400:  # >= 1 day
            eta_days = int(eta_seconds / 86400)
            eta_hours = int((eta_seconds % 86400) / 3600)
            return f"{eta_days}d{eta_hours}h"
        elif eta_seconds >= 3600:  # >= 1 hour
            eta_hours = int(eta_seconds / 3600)
            eta_minutes = int((eta_seconds % 3600) / 60)
            return f"{eta_hours}h{eta_minutes:02d}m"
        elif eta_seconds > 60:
            eta_minutes = int(eta_seconds / 60)
            eta_seconds = int(eta_seconds % 60)
            return f"{eta_minutes}m{eta_seconds:02d}s"
        else:
            return f"{int(eta_seconds)}s"

    def _create_progress_bar_string(self, current: int, total: int, total_found: int = 0, prefix: str = "") -> str:
        """Create a progress bar string with current/total counts, found count, and ETA."""
        import shutil
        
        progress = current / total if total > 0 else 0
        
        # Calculate and add ETA if we have enough data
        eta_text = ""
        if self.start_time and current > self.session_start_processed:
            import time
            elapsed = time.time() - self.start_time
            session_work_done = current - self.session_start_processed
            if session_work_done > 0:
                avg_time_per_item = elapsed / session_work_done
                remaining_items = total - current
                eta_seconds = remaining_items * avg_time_per_item
                eta_text = f" | ETA: {self._format_eta(eta_seconds)}"
        
        # Create compact format (used as fallback and for space calculation)
        found_text = f" | Found: {total_found}" if total_found > 0 else ""
        compact_status = f"{prefix}[{current:4d}/{total}] {progress:.1%}" + found_text + eta_text
        
        # Try to add progress bar if we have terminal width and enough space
        try:
            terminal_width = shutil.get_terminal_size().columns
            base_status = f"{prefix}[{current:4d}/{total}] "
            percentage_text = f" {progress:.1%}"
            fixed_content_length = len(base_status + percentage_text + found_text + eta_text)
            available_for_bar = terminal_width - fixed_content_length
            
            # Use progress bar if we have at least 10 characters for it
            if available_for_bar >= 10:
                bar_length = min(50, available_for_bar)  # Use available space, max 50
                filled_length = int(bar_length * progress)
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                return base_status + bar + percentage_text + found_text + eta_text
            else:
                # Not enough space for progress bar, use compact format
                return compact_status
        except:
            # If we can't get terminal size, use compact format
            return compact_status

    def _update_progress(self, current: int, total: int, total_found: int = 0):
        """Update progress bar in place."""
        if not self.verbose:
            status = self._create_progress_bar_string(current, total, total_found)
            
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

    def _process_paragraph_batch(self, paragraph_data: List[Tuple[int, str]], batch_size: int = 4) -> List[Tuple[int, str, float, str, str]]:
        """Process a batch of paragraphs in parallel."""
        results: List[Tuple[int, str, float, str, str]] = []
        
        def process_single_paragraph(paragraph_info: Tuple[int, str]) -> Tuple[int, str, float, str, str]:
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
                        paragraph_num, response, _, _, _ = result
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

    def _process_paragraphs_continuously(self, paragraph_list: List[Tuple[int, str]], paragraphs: List[str], content_type: str, total_paragraphs: int, verbose: bool = False) -> None:
        """Process all paragraphs continuously with up to 8 concurrent evaluations."""
        import threading
        from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
        
        # Thread-safe state management with scrolling window
        stanza_states: Dict[int, Dict[str, Any]] = {}
        completed_stanzas: set[int] = set()  # Track which stanzas are completed
        active_processes: Dict[int, str] = {}  # Currently processing stanzas
        display_lock: threading.Lock = threading.Lock()
        
        # Scrolling window for display (show last N completed + active processes)
        MAX_COMPLETED_DISPLAY = 8  # Show only last 8 completed items
        recent_completed: List[int] = []  # Recently completed stanzas in order
        display_initialized = False  # Track if we've shown the display yet
        
        # Initialize all paragraph states
        for paragraph_num, paragraph in paragraph_list:
            stanza_states[paragraph_num] = {
                'paragraph': paragraph,
                'decisions': [],
                'final_result': None
            }
        
        def update_display() -> None:
            """Update the scrolling display: recent completions + active processes."""
            nonlocal display_initialized
            if not verbose:
                return
                
            with display_lock:
                if display_initialized:
                    # Clear the entire display area first
                    # Total lines: 1 (header) + 8 (completed) + 1 (header) + 8 (active) + 1 (progress bar) = 19 lines
                    total_lines = 1 + MAX_COMPLETED_DISPLAY + 1 + 8 + 1
                    
                    # Move cursor up to overwrite previous display
                    sys.stdout.write(f"\033[{total_lines}A")
                    
                    # Clear from cursor to end of screen to avoid artifacts
                    sys.stdout.write("\033[0J")
                else:
                    # First time - just start writing
                    display_initialized = True
                
                # Update recent completions section
                sys.stdout.write("Recent Completions:\n")
                
                for i in range(MAX_COMPLETED_DISPLAY):
                    if i < len(recent_completed):
                        stanza_num = recent_completed[-(i+1)]  # Show most recent first
                        state = stanza_states[stanza_num]
                        decisions_str = " | ".join(state['decisions'])
                        response, confidence, _ = state['final_result']
                        result_status = f"● found (conf: {confidence:.2f})" if response == self.positive_label else f"○ skip (conf: {confidence:.2f})"
                        sys.stdout.write(f"  Stanza {stanza_num + 1}/{total_paragraphs}: {decisions_str} → {result_status}\n")
                    else:
                        sys.stdout.write("  [waiting...]\n")
                
                # Update active processes section
                sys.stdout.write("Active Processes:\n")
                
                active_list = list(active_processes.items())
                for i in range(8):
                    if i < len(active_list):
                        stanza_num, status = active_list[i]
                        sys.stdout.write(f"  Stanza {stanza_num + 1}/{total_paragraphs}: {status}\n")
                    else:
                        sys.stdout.write("  [waiting...]\n")
                
                # Add progress bar at the bottom (using the shared helper method)
                progress_bar_string = self._create_progress_bar_string(
                    self.current_total_processed, 
                    total_paragraphs, 
                    self.current_total_found, 
                    "Progress: "
                )
                sys.stdout.write(progress_bar_string + "\n")
                sys.stdout.flush()
        
        def add_completed_stanza(stanza_num: int) -> None:
            """Add a stanza to the completed list, maintaining the scrolling window."""
            recent_completed.append(stanza_num)
            # Keep only the most recent items
            if len(recent_completed) > MAX_COMPLETED_DISPLAY:
                recent_completed.pop(0)

        with ThreadPoolExecutor(max_workers=8) as executor:
            active_futures: Dict[Any, Tuple[int, int]] = {}
            paragraph_queue: List[Tuple[int, str]] = list(paragraph_list)
            
            def start_next_stanza_if_available() -> bool:
                """Start the next stanza from queue if available. Returns True if started."""
                if paragraph_queue and self.model_names:
                    try:
                        next_paragraph_num, next_paragraph = paragraph_queue.pop(0)
                        next_future = executor.submit(self.classify_text, next_paragraph, self.model_names[0], is_final_model=False)
                        active_futures[next_future] = (next_paragraph_num, 0)
                        # Update active processes display
                        active_processes[next_paragraph_num] = "Model 1: processing..."
                        update_display()
                        return True
                    except Exception as e:
                        if self.verbose:
                            print(f"\nError starting next stanza: {e}")
                        return False
                return False
            
            # Start initial batch of Model 1 evaluations (up to 8)
            for _ in range(min(8, len(paragraph_queue))):
                start_next_stanza_if_available()
            
            # Process results as they complete and keep queue flowing
            while active_futures and not self.cancelled:
                done, _ = wait(active_futures.keys(), return_when=FIRST_COMPLETED)
                
                for future in done:
                    paragraph_num, model_idx = active_futures.pop(future)
                    
                    # Skip processing if cancelled
                    if self.cancelled:
                        continue
                    
                    if paragraph_num in completed_stanzas:
                        continue
                    
                    try:
                        response, confidence, explanation = future.result()
                        state = stanza_states[paragraph_num]
                        state['decisions'].append(f"Model {model_idx + 1}: {response}")
                        decisions_str = " | ".join(state['decisions'])
                        
                        if response == self.negative_label:
                            # Rejected - finalize this stanza
                            state['decisions'][-1] += " → rejected"
                            state['final_result'] = (response, confidence, "")
                            
                            # Update progress tracking
                            self.current_paragraph = paragraph_num
                            self.current_total_processed += 1
                            self.completed_paragraphs.add(paragraph_num)
                            
                            # Move to completed and remove from active
                            completed_stanzas.add(paragraph_num)
                            if paragraph_num in active_processes:
                                del active_processes[paragraph_num]
                            
                            # Update display for both verbose and non-verbose
                            if verbose:
                                # Add to recent completed and update display
                                add_completed_stanza(paragraph_num)
                                update_display()
                            else:
                                # Simple progress bar for non-verbose mode
                                processed_count = self.current_total_processed
                                found_count = self.current_total_found
                                self._update_progress(processed_count, total_paragraphs, found_count)
                            
                            # Save progress
                            progress: Dict[str, Any] = {
                                "completed_paragraphs": list(self.completed_paragraphs),
                                "total_found": self.current_total_found
                            }
                            self.save_progress(progress)
                            
                            # Start next paragraph
                            start_next_stanza_if_available()
                            
                        elif model_idx + 1 < len(self.model_names):
                            # Continue to next model
                            next_model_status = f"{decisions_str} | Model {model_idx + 2}: processing..."
                            active_processes[paragraph_num] = next_model_status
                            update_display()
                            
                            # Submit next model
                            is_final_model = (model_idx + 1 == len(self.model_names) - 1)
                            next_future = executor.submit(
                                self.classify_text, 
                                state['paragraph'], 
                                self.model_names[model_idx + 1], 
                                is_final_model=is_final_model
                            )
                            active_futures[next_future] = (paragraph_num, model_idx + 1)
                            
                            # Start new stanza
                            start_next_stanza_if_available()
                            
                        else:
                            # All models approved - finalize
                            state['decisions'].append(f"→ All {len(self.model_names)} approved")
                            state['final_result'] = (response, 0.95, explanation)
                            
                            # Update progress tracking
                            self.current_paragraph = paragraph_num
                            self.current_total_processed += 1
                            self.current_total_found += 1
                            self.completed_paragraphs.add(paragraph_num)
                            self.append_result(paragraph_num + 1, paragraphs[paragraph_num], 0.95, explanation)
                            
                            # Update progress display
                            if verbose:
                                # Move to completed and remove from active
                                completed_stanzas.add(paragraph_num)
                                if paragraph_num in active_processes:
                                    del active_processes[paragraph_num]
                                
                                # Add to recent completed and update display
                                add_completed_stanza(paragraph_num)
                                update_display()
                            else:
                                # Simple progress bar for non-verbose mode
                                processed_count = self.current_total_processed
                                found_count = self.current_total_found
                                self._update_progress(processed_count, total_paragraphs, found_count)
                            
                            # Save progress (single save for both verbose and non-verbose)
                            progress: Dict[str, Any] = {
                                "completed_paragraphs": list(self.completed_paragraphs),
                                "total_found": self.current_total_found
                            }
                            self.save_progress(progress)
                            
                            # Start next paragraph
                            start_next_stanza_if_available()
                        
                    except Exception as e:
                        if self.verbose:
                            print(f"\nError processing stanza {paragraph_num}, model {model_idx + 1}: {e}")
                        
                        state = stanza_states[paragraph_num]
                        state['final_result'] = ('0', 0.0, "")
                        self.current_total_processed += 1
                        self.completed_paragraphs.add(paragraph_num)
                        completed_stanzas.add(paragraph_num)
                        if paragraph_num in active_processes:
                            del active_processes[paragraph_num]
                        
                        # Update display for both verbose and non-verbose
                        if verbose:
                            # Add to recent completed and update display
                            add_completed_stanza(paragraph_num)
                            update_display()
                        else:
                            # Simple progress bar for non-verbose mode
                            processed_count = self.current_total_processed
                            found_count = self.current_total_found
                            self._update_progress(processed_count, total_paragraphs, found_count)
                        
                        start_next_stanza_if_available()


def verify_models_available(model_names: Union[str, List[str]]) -> None:
    """Verify that the specified model(s) are available in Ollama."""
    models_response = ollama.list()
    
    # Extract model names from the response
    available_models: List[str] = [model.model for model in models_response.models if model.model]
    
    print(f"Available models: {available_models}")
    
    # Support both single model (backward compatibility) and multiple models
    if isinstance(model_names, str):
        models_to_check: List[str] = [model_names]
    else:
        models_to_check = model_names
    
    missing_models: List[str] = []
    for model_name in models_to_check:
        if model_name not in available_models:
            missing_models.append(model_name)
    
    if missing_models:
        missing_str: str = ', '.join(missing_models)
        available_str: str = ', '.join(available_models)
        pull_commands: str = ' && '.join([f"ollama pull {model}" for model in missing_models])
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract matching paragraphs from text using a local LLM")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed debug output")
    parser.add_argument("--config", "-c", help="Configuration file path")
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
classifier_instance: Optional[Any] = None

def signal_handler(signum: int, frame: Any) -> None:
    """Handle Ctrl+C by saving progress and terminating immediately."""
    global classifier_instance
    
    print("\n\nReceived cancellation signal (Ctrl+C). Saving progress and stopping immediately...")
    
    # Set cancellation flag to stop processing loop
    if classifier_instance is not None:
        classifier_instance.cancelled = True
    
    # Save current progress - classifier_instance is guaranteed to exist
    try:
        if classifier_instance is not None:
            current_progress: Dict[str, Any] = {
                "completed_paragraphs": list(classifier_instance.completed_paragraphs),
                "total_found": classifier_instance.current_total_found
            }
            classifier_instance.save_progress(current_progress)
            total_processed = len(classifier_instance.completed_paragraphs)
            print(f"Progress saved (processed: {total_processed}, found: {current_progress['total_found']}).")
        else:
            print("No classifier instance to save progress from.")
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
