# Document Processing Pipeline

A Python tool that converts various document formats to a combined Markdown file with optional metadata, tagging, and OpenAI-powered summarization.

## Features

- Processes PDF, DOCX, TXT, HTML, and CSV files
- Converts all formats to Markdown while preserving structure
- Adds metadata and tags as YAML front matter
- Optionally generates summaries using OpenAI API
- Combines all content into a single output file

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/datapipeline.git
cd datapipeline

# Install dependencies
pip install -r requirements.txt
```

## Usage

Place files to be processed in the same directory as the script, then run:

```bash
# Basic usage (processes all supported files in current directory)
python pipeline.py

# Add metadata
python pipeline.py --metadata "author=Jane Doe" "project=Research"

# Add tags
python pipeline.py --tags research documentation reference

# Generate summaries with OpenAI
python pipeline.py --summary --model gpt-4 --length 200

# Customize output file
python pipeline.py --output combined_docs.md

# All options combined
python pipeline.py --metadata "author=Jane Doe" "project=Research" \
                 --tags research documentation reference \
                 --summary --model gpt-3.5-turbo --length 150 \
                 --output combined_docs.md
```

## Requirements

- Python 3.8+
- OpenAI API key (set as OPENAI_API_KEY environment variable) for summarization
- Dependencies listed in requirements.txt

## License

MIT