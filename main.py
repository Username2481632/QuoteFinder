#!/usr/bin/env python3
"""
Social Comparison Finder - Extract social comparison paragraphs from text using a local LLM

This script processes HTML/text files paragraph by paragraph, sends each paragraph
to a local language model via Ollama, and logs paragraphs that show people
comparing themselves to others.

Requirements:
- Ollama installed and running
- DeepSeek-R1 distill model pulled in Ollama
- Python packages: beautifulsoup4, ollama

Usage:
    python pdf_quote_finder.py
"""

import json
import re
import sys
import subprocess
import time
from pathlib import Path
from typing import List, Tuple, Dict
from bs4 import BeautifulSoup
import ollama


class SocialComparisonFinder:
    def __init__(self, model_name: str = "deepseek-r1:1.5b"):
        """
        Initialize the Social Comparison Finder.
        
        Args:
            model_name: Name of the Ollama model to use
        """
        self.model_name = model_name
        self.prompt_template = self._get_hardcoded_prompt()
        self.progress_file = "progress.json"
        self.output_file = "results.txt"
        self.results_count = 0
        
    def _get_hardcoded_prompt(self) -> str:
        """Get the hardcoded prompt for finding social comparison paragraphs."""
        return """Please evaluate this paragraph from a book. Respond with ONLY "1" if the paragraph contains someone comparing themselves to another person, such as:
- Feeling inferior or superior to someone else
- Measuring their abilities, looks, success, or traits against another person
- Expressing envy, jealousy, or admiration based on comparison
- Self-doubt triggered by seeing someone else's qualities
- Competitive thoughts about being better or worse than someone
- Social comparison of any kind (appearance, intelligence, status, achievements, etc.)

Respond with ONLY "0" if the paragraph does NOT contain clear self-comparison to others, such as:
- General description without comparison
- Internal thoughts not related to comparing with others
- Simple dialogue or narrative
- Observations about others without self-comparison

You must respond with only the number 1 or 0, absolutely nothing else.

Paragraph: {paragraph}"""
    def load_progress(self) -> Dict[str, int]:
        """Load progress from JSON file."""
        try:
            if Path(self.progress_file).exists():
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load progress file: {e}")
        return {"last_paragraph": 0, "total_processed": 0, "total_found": 0}
    
    def save_progress(self, progress: Dict[str, int]):
        """Save progress to JSON file."""
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save progress: {e}")
    
    def _create_output_file_header(self):
        """Create output file with header if it doesn't exist."""
        if not Path(self.output_file).exists():
            try:
                with open(self.output_file, 'w', encoding='utf-8') as f:
                    f.write("Social Comparison Paragraphs Found\n")
                    f.write("=" * 50 + "\n\n")
            except Exception as e:
                print(f"Error creating output file: {e}")
    
    def append_result(self, paragraph_num: int, text: str, confidence: float):
        """Append a result immediately to the output file."""
        try:
            # Create header if this is the first result
            if self.results_count == 0:
                with open(self.output_file, 'w', encoding='utf-8') as f:
                    f.write("Social Comparison Paragraphs Found\n")
                    f.write("=" * 50 + "\n\n")
            
            with open(self.output_file, 'a', encoding='utf-8') as f:
                self.results_count += 1
                f.write(f"Finding #{self.results_count} (Paragraph {paragraph_num}, Confidence: {confidence:.2f})\n")
                f.write("-" * 60 + "\n")
                f.write(f"{text}\n\n")
            
            print(f"✓ SOCIAL COMPARISON FOUND (confidence: {confidence:.2f}) - Saved to {self.output_file}")
            
        except Exception as e:
            print(f"Error saving result: {e}")

    def _extract_text_from_html(self, html_path: str) -> str:
        """Extract clean text from HTML file."""
        try:
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
            
        except Exception as e:
            print(f"Error extracting text from HTML: {e}")
            raise

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

    def _query_model(self, paragraph: str) -> tuple[str, float]:
        """Send paragraph to the language model and get response with confidence."""
        try:
            prompt = self.prompt_template.format(paragraph=paragraph)
            
            # Generate response using ollama.chat
            response = ollama.chat(
                model=self.model_name,
                messages=[{
                    'role': 'user',
                    'content': prompt
                }],
                options={
                    'temperature': 0.1,
                    'top_p': 0.9,
                    'num_predict': 10,    # Allow a few tokens for response
                    'num_ctx': 4096,     # Context window
                },
                stream=False
            )
            
            # Get the generated text
            generated_text = response['message']['content'].strip()
            
            # Parse the response for 1 or 0
            if '1' in generated_text and '0' not in generated_text:
                return '1', 0.9  # High confidence for clear "1"
            elif '0' in generated_text and '1' not in generated_text:
                return '0', 0.9  # High confidence for clear "0"
            elif '1' in generated_text:
                return '1', 0.7  # Medium confidence if both present, prefer 1
            elif '0' in generated_text:
                return '0', 0.7  # Medium confidence if both present, prefer 0
            else:
                print(f"Warning: Unexpected model response: '{generated_text}' - treating as 0")
                return '0', 0.3  # Low confidence for unclear response
                
        except Exception as e:
            print(f"Error querying model: {e}")
            return '0', 0.0  # No confidence on error

    def _query_model_with_probabilities(self, paragraph: str) -> tuple[str, float]:
        """
        Get model's preference between "1" and "0" by comparing their probabilities.
        This is more reliable than parsing text responses.
        """
        prompt = self.prompt_template.format(paragraph=paragraph)
        
        try:
            # Method 1: Compare probabilities by testing both completions
            prob_1 = self._evaluate_completion_likelihood(prompt, "1")
            prob_0 = self._evaluate_completion_likelihood(prompt, "0")
            
            # Normalize probabilities
            total_prob = prob_1 + prob_0
            if total_prob > 0:
                norm_prob_1 = prob_1 / total_prob
                norm_prob_0 = prob_0 / total_prob
                
                if norm_prob_1 > norm_prob_0:
                    return '1', norm_prob_1
                else:
                    return '0', norm_prob_0
            else:
                # Fallback to regular generation
                return self._fallback_query(prompt)
                
        except Exception as e:
            print(f"Error in probability-based query: {e}")
            return self._fallback_query(prompt)

    def _evaluate_completion_likelihood(self, prompt: str, completion: str) -> float:
        """
        Evaluate how likely the model thinks a completion is.
        Uses a technique where we measure the model's "confidence" in generating
        the completion by looking at repeated sampling.
        """
        try:
            # Sample multiple times with low temperature to see consistency
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
                        'temperature': 0.3,  # Some randomness but not too much
                        'top_p': 0.9,
                        'num_predict': 1,
                    }
                )
                
                generated = response['message']['content'].strip()
                if completion in generated:
                    matches += 1
            
            # Return proportion of matches as probability estimate
            return matches / total_samples
            
        except:
            return 0.5  # Default probability

    def _fallback_query(self, prompt: str) -> tuple[str, float]:
        """Fallback method using standard text generation."""
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{
                    'role': 'user',
                    'content': prompt
                }],
                options={
                    'temperature': 0.1,
                    'num_predict': 1,
                }
            )
            
            content = response['message']['content'].strip()
            if '1' in content:
                return '1', 0.8
            elif '0' in content:
                return '0', 0.8
            else:
                return '0', 0.5
                
        except Exception as e:
            print(f"Error in fallback query: {e}")
            return '0', 0.0

    def process_text_file(self, text_path: str = "huckleberry_finn.html") -> Dict[str, int]:
        """
        Process the text file and return results.
        
        Args:
            text_path: Path to the HTML text file
            
        Returns:
            Dictionary with processing statistics
        """
        text_file = Path(text_path)
        if not text_file.exists():
            raise FileNotFoundError(f"Text file not found: {text_file}")

        print(f"Processing text file: {text_file}")
        print(f"Using model: {self.model_name}")
        
        # Load progress
        progress = self.load_progress()
        start_paragraph = progress["last_paragraph"]
        total_processed = progress["total_processed"]
        total_found = progress["total_found"]
        
        print(f"Resuming from paragraph {start_paragraph + 1}")
        if total_found > 0:
            print(f"Previously found {total_found} social comparisons")
        
        # Always create output file header
        self._create_output_file_header()
        
        try:
            # Extract text from HTML
            full_text = self._extract_text_from_html(str(text_file))
            
            # Split into paragraphs
            paragraphs = self._split_into_paragraphs(full_text)
            total_paragraphs = len(paragraphs)
            
            print(f"Total paragraphs to process: {total_paragraphs}")
            print(f"Starting from paragraph: {start_paragraph + 1}")
            
            for i, paragraph in enumerate(paragraphs[start_paragraph:], start_paragraph):
                paragraph_num = i + 1
                print(f"Processing paragraph {paragraph_num}/{total_paragraphs}: ", end="", flush=True)
                
                # Query the model
                response, confidence = self._query_model_with_probabilities(paragraph)
                
                total_processed += 1
                
                if response == '1':
                    total_found += 1
                    self.append_result(paragraph_num, paragraph, confidence)
                else:
                    print(f"○ skipped (confidence: {confidence:.2f})")
                
                # Save progress after each paragraph
                progress = {
                    "last_paragraph": i + 1,
                    "total_processed": total_processed,
                    "total_found": total_found
                }
                self.save_progress(progress)
                
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
        try:
            with open(self.output_file, 'a', encoding='utf-8') as f:
                f.write("\n" + "=" * 50 + "\n")
                f.write("PROCESSING COMPLETE\n")
                f.write("=" * 50 + "\n")
                f.write(f"Total paragraphs processed: {total_processed}\n")
                f.write(f"Social comparisons found: {total_found}\n")
                if total_processed > 0:
                    f.write(f"Success rate: {total_found/total_processed*100:.1f}%\n")
        except Exception as e:
            print(f"Error writing completion summary: {e}")
        
        # Clear progress file on completion
        try:
            Path(self.progress_file).unlink(missing_ok=True)
        except:
            pass
        
        return progress


def ensure_ollama_running():
    """Ensure Ollama is running, start it if not."""
    try:
        # Test if Ollama is responding
        ollama.list()
        return True
    except Exception:
        print("Ollama not running. Starting Ollama...")
        try:
            # Start Ollama in the background
            subprocess.Popen(['ollama', 'serve'], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
            
            # Wait for Ollama to start
            for i in range(10):  # Wait up to 10 seconds
                time.sleep(1)
                try:
                    ollama.list()
                    print("Ollama started successfully!")
                    return True
                except:
                    continue
            
            print("Failed to start Ollama after 10 seconds")
            return False
            
        except FileNotFoundError:
            print("Ollama not found. Please install Ollama first.")
            print("Visit: https://ollama.ai/download")
            return False
        except Exception as e:
            print(f"Error starting Ollama: {e}")
            return False


def main():
    print("Social Comparison Finder")
    print("=" * 30)
    
    # Ensure Ollama is running
    if not ensure_ollama_running():
        sys.exit(1)
    
    # Check if model is available
    model_name = "deepseek-r1:1.5b"
    try:
        models_response = ollama.list()
        print(f"Debug: Ollama response: {models_response}")
        
        # Handle different response formats
        if isinstance(models_response, dict) and 'models' in models_response:
            model_names = [model.get('name', model.get('model', '')) for model in models_response['models']]
        else:
            # Try a simple test call instead
            print("Trying test call to verify Ollama connection...")
            test_response = ollama.chat(
                model=model_name, 
                messages=[{'role': 'user', 'content': 'test'}], 
                options={'num_predict': 1}
            )
            print("Ollama connection successful!")
            model_names = [model_name]  # Assume model exists if test call works
        
        if model_name not in model_names:
            print(f"Error: Model '{model_name}' not found in Ollama.")
            print(f"Available models: {', '.join(model_names)}")
            print(f"Run: ollama pull {model_name}")
            sys.exit(1)
        else:
            print(f"Using model: {model_name}")
            
    except Exception as e:
        print(f"Error checking model: {e}")
        print(f"Error type: {type(e)}")
        print(f"Run: ollama pull {model_name}")
        sys.exit(1)
    
    # Create finder and process text file
    finder = SocialComparisonFinder(model_name=model_name)
    
    try:
        results = finder.process_text_file("huckleberry_finn.html")
        print(f"\nResults saved to: {finder.output_file}")
        print(f"Progress file: {finder.progress_file}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
