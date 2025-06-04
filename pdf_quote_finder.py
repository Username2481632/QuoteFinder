#!/usr/bin/env python3
"""
PDF Quote Finder - Extract interesting paragraphs from PDF books using a local LLM

This script processes a PDF book paragraph by paragraph, sends each paragraph
to a local language model via Ollama, and logs paragraphs that the model
rates as "1" (interesting/relevant) to an output file.

Requirements:
- Ollama installed and running
- DeepSeek-R1 distill model pulled in Ollama
- Python packages: pdfplumber, ollama

Usage:
    python pdf_quote_finder.py path/to/book.pdf
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Tuple
import pdfplumber
import ollama


class PDFQuoteFinder:
    def __init__(self, model_name: str = "deepseek-r1-distill-llama-8b", 
                 prompt_template: str = None):
        """
        Initialize the PDF Quote Finder.
        
        Args:
            model_name: Name of the Ollama model to use
            prompt_template: Custom prompt template (uses default if None)
        """
        self.model_name = model_name
        self.prompt_template = prompt_template or self._get_default_prompt()
        self.results = []
        
    def _get_default_prompt(self) -> str:
        """Get the default prompt template for evaluating paragraphs."""
        return """Please evaluate this paragraph from a book. Respond with ONLY "1" if the paragraph contains:
- Interesting quotes or insights
- Memorable passages worth highlighting
- Important concepts or ideas
- Thought-provoking statements

Respond with ONLY "0" if the paragraph is:
- Mundane description or narrative
- Not particularly noteworthy
- Routine dialogue or exposition

Respond with only the number 1 or 0, nothing else.

Paragraph: {paragraph}"""

    def _clean_paragraph(self, text: str) -> str:
        """Clean and normalize paragraph text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        # Remove page numbers and headers/footers (simple heuristic)
        text = re.sub(r'^\d+\s*$', '', text)
        return text

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        # Split by double newlines or single newlines followed by indentation
        paragraphs = re.split(r'\n\s*\n|\n(?=\s{4,})', text)
        
        # Clean and filter paragraphs
        cleaned_paragraphs = []
        for para in paragraphs:
            cleaned = self._clean_paragraph(para)
            # Skip very short paragraphs (likely page numbers, headers, etc.)
            if len(cleaned) > 50:  # Minimum paragraph length
                cleaned_paragraphs.append(cleaned)
        
        return cleaned_paragraphs

    def _query_model(self, paragraph: str) -> str:
        """Send paragraph to the language model and get response."""
        try:
            prompt = self.prompt_template.format(paragraph=paragraph)
            
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                options={
                    'temperature': 0.1,  # Low temperature for consistent responses
                    'top_p': 0.9,
                    'num_predict': 5,    # We only need 1 character response
                }
            )
            
            # Extract just the content and clean it
            content = response['message']['content'].strip()
            # Look for "1" or "0" in the response
            if '1' in content:
                return '1'
            elif '0' in content:
                return '0'
            else:
                print(f"Warning: Unexpected model response: {content}")
                return '0'  # Default to 0 if unclear
                
        except Exception as e:
            print(f"Error querying model: {e}")
            return '0'  # Default to 0 on error

    def process_pdf(self, pdf_path: str) -> List[dict]:
        """
        Process the entire PDF and return results.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of dictionaries containing interesting paragraphs
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        print(f"Processing PDF: {pdf_path}")
        print(f"Using model: {self.model_name}")
        
        results = []
        total_paragraphs = 0
        interesting_paragraphs = 0
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                print(f"Total pages: {total_pages}")
                
                for page_num, page in enumerate(pdf.pages, 1):
                    print(f"Processing page {page_num}/{total_pages}...")
                    
                    # Extract text from the page
                    page_text = page.extract_text()
                    if not page_text:
                        continue
                    
                    # Split into paragraphs
                    paragraphs = self._split_into_paragraphs(page_text)
                    
                    for paragraph in paragraphs:
                        total_paragraphs += 1
                        print(f"  Paragraph {total_paragraphs}: ", end="", flush=True)
                        
                        # Query the model
                        response = self._query_model(paragraph)
                        
                        if response == '1':
                            interesting_paragraphs += 1
                            result = {
                                'page': page_num,
                                'paragraph_number': total_paragraphs,
                                'text': paragraph,
                                'timestamp': str(Path().resolve())  # Could use datetime if needed
                            }
                            results.append(result)
                            print("✓ INTERESTING")
                        else:
                            print("○ skipped")
                
        except Exception as e:
            print(f"Error processing PDF: {e}")
            raise
        
        print(f"\nProcessing complete!")
        print(f"Total paragraphs processed: {total_paragraphs}")
        print(f"Interesting paragraphs found: {interesting_paragraphs}")
        print(f"Success rate: {interesting_paragraphs/total_paragraphs*100:.1f}%")
        
        return results

    def save_results(self, results: List[dict], output_path: str):
        """Save results to a file."""
        output_path = Path(output_path)
        
        # Save as JSON for structured data
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Also save as readable text file
        txt_path = output_path.with_suffix('.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"Interesting Paragraphs Found: {len(results)}\n")
            f.write("=" * 50 + "\n\n")
            
            for i, result in enumerate(results, 1):
                f.write(f"Quote #{i} (Page {result['page']})\n")
                f.write("-" * 30 + "\n")
                f.write(f"{result['text']}\n\n")
        
        print(f"Results saved to:")
        print(f"  - {json_path} (structured data)")
        print(f"  - {txt_path} (readable format)")


def main():
    parser = argparse.ArgumentParser(description="Extract interesting paragraphs from PDF using local LLM")
    parser.add_argument("pdf_path", help="Path to the PDF file to process")
    parser.add_argument("-o", "--output", default="interesting_quotes", 
                      help="Output file prefix (default: interesting_quotes)")
    parser.add_argument("-m", "--model", default="deepseek-r1-distill-llama-8b",
                      help="Ollama model name to use")
    parser.add_argument("--prompt", help="Custom prompt template file")
    
    args = parser.parse_args()
    
    # Load custom prompt if provided
    custom_prompt = None
    if args.prompt:
        try:
            with open(args.prompt, 'r') as f:
                custom_prompt = f.read().strip()
        except Exception as e:
            print(f"Error loading custom prompt: {e}")
            sys.exit(1)
    
    # Check if Ollama is running and model is available
    try:
        models = ollama.list()
        model_names = [model['name'] for model in models['models']]
        if args.model not in model_names:
            print(f"Error: Model '{args.model}' not found in Ollama.")
            print(f"Available models: {', '.join(model_names)}")
            print(f"Run: ollama pull {args.model}")
            sys.exit(1)
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        print("Make sure Ollama is installed and running.")
        print("Visit: https://ollama.ai/download")
        sys.exit(1)
    
    # Create finder and process PDF
    finder = PDFQuoteFinder(model_name=args.model, prompt_template=custom_prompt)
    
    try:
        results = finder.process_pdf(args.pdf_path)
        finder.save_results(results, args.output)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
