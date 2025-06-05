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
from pathlib import Path
from typing import List, Tuple, Dict
from bs4 import BeautifulSoup
import ollama


class TextClassifier:
    def __init__(self, model_name: str = "deepseek-r1:1.5b"):
        """
        Initialize the Text Classifier.
        
        Args:
            model_name: Name of the Ollama model to use
        """
        self.model_name = model_name
        self.prompt_template = self._get_classification_prompt()
        self.progress_file = "progress.json"
        self.output_file = "results.txt"
        self.results_count = 0
        self.positive_label = "1"
        self.negative_label = "0"
        
    def _get_classification_prompt(self) -> str:
        """Get the classification prompt. This can be easily modified for different tasks."""
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
        if Path(self.progress_file).exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {"last_paragraph": 0, "total_processed": 0, "total_found": 0}
    
    def save_progress(self, progress: Dict[str, int]):
        """Save progress to JSON file."""
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
    
    def _create_output_file_header(self):
        """Create output file with header if it doesn't exist."""
        if not Path(self.output_file).exists():
            with open(self.output_file, 'w', encoding='utf-8') as f:
                f.write("Social Comparison Paragraphs Found\n")
                f.write("=" * 50 + "\n\n")
    
    def append_result(self, paragraph_num: int, text: str, confidence: float):
        """Append a result immediately to the output file."""
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

    def classify_text(self, text: str) -> tuple[str, float]:
        """Classify text and return label with confidence."""
        prompt = self.prompt_template.format(paragraph=text)
        
        response = ollama.chat(
            model=self.model_name,
            messages=[{
                'role': 'user',
                'content': prompt
            }],
            options={
                'temperature': 0.1,
                'top_p': 0.9,
                'num_predict': 10,
                'num_ctx': 4096,
            },
            stream=False
        )
        
        generated_text = response['message']['content'].strip()
        
        # Handle DeepSeek-R1 thinking format - extract final answer after </think>
        if '</think>' in generated_text:
            # Get text after the thinking tags
            final_answer = generated_text.split('</think>')[-1].strip()
        else:
            final_answer = generated_text
        
        # Parse the response for clear answers
        if final_answer == self.positive_label:
            return self.positive_label, 0.95
        elif final_answer == self.negative_label:
            return self.negative_label, 0.95
        elif self.positive_label in final_answer and self.negative_label not in final_answer:
            return self.positive_label, 0.85
        elif self.negative_label in final_answer and self.positive_label not in final_answer:
            return self.negative_label, 0.85
        else:
            # Unclear response - use probability comparison
            print(f"Unclear response: '{final_answer}' - checking probabilities...")
            return self._resolve_by_probability(prompt)
    
    def _resolve_by_probability(self, prompt: str) -> tuple[str, float]:
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
            return self.positive_label, norm_prob_positive
        else:
            return self.negative_label, norm_prob_negative
    
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
                    'num_predict': 10,  # Allow more tokens for thinking
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
                response, confidence = self.classify_text(paragraph)
                
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
        with open(self.output_file, 'a', encoding='utf-8') as f:
            f.write("\n" + "=" * 50 + "\n")
            f.write("PROCESSING COMPLETE\n")
            f.write("=" * 50 + "\n")
            f.write(f"Total paragraphs processed: {total_processed}\n")
            f.write(f"Social comparisons found: {total_found}\n")
            if total_processed > 0:
                f.write(f"Success rate: {total_found/total_processed*100:.1f}%\n")
        
        # Clear progress file on completion
        Path(self.progress_file).unlink(missing_ok=True)
        
        return progress


def verify_model_available(model_name: str):
    """Verify that the specified model is available in Ollama."""
    models_response = ollama.list()
    
    # Extract model names from the response
    model_names = [model.model for model in models_response.models]
    
    print(f"Available models: {model_names}")
    
    if model_name not in model_names:
        raise ValueError(f"Model '{model_name}' not found. Available models: {', '.join(model_names)}. Run: ollama pull {model_name}")
    
    print(f"Using model: {model_name}")


def main():
    print("Text Classifier")
    print("=" * 30)
    
    # Check if model is available
    model_name = "deepseek-r1:1.5b"
    verify_model_available(model_name)
    
    # Test the model
    print("Trying test call to verify Ollama connection...")
    ollama.chat(
        model=model_name, 
        messages=[{'role': 'user', 'content': 'test'}], 
        options={'num_predict': 1}
    )
    print("Ollama connection successful!")
    
    # Create classifier and process text file
    classifier = TextClassifier(model_name=model_name)
    classifier.process_text_file("huckleberry_finn.html")
    
    print(f"\nResults saved to: {classifier.output_file}")
    print(f"Progress file: {classifier.progress_file}")


if __name__ == "__main__":
    main()
