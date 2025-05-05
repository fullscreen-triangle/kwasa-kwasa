# ScienceMedium: Intelligent Scientific Content Publishing Pipeline

ScienceMedium is an AI-powered writing and publishing pipeline designed specifically for researchers and science writers. This tool addresses the unique challenges of scientific writing by combining intelligent research assistance, flexible content creation, and automated publishing to Medium.

## Overview

ScienceMedium eliminates the friction between complex scientific content creation and online publishing by:

1. **Format-Agnostic Writing**: Write in your preferred style without rigid Markdown/LaTeX constraints
2. **On-Demand AI Assistance**: Get intelligent writing suggestions only when you request them
3. **Metacognitive Orchestration**: An LLM-powered system that ensures quality and completeness
4. **Contextual Research Integration**: Automated information gathering tied to your specific context
5. **Seamless Medium Publishing**: Automatic conversion and publishing once content meets quality standards

## Features

### 🖋️ Format-Agnostic Writing
- Write naturally without worrying about specific Markdown or LaTeX syntax
- The system handles conversion to Medium-compatible format automatically
- Support for scientific notation, formulas, tables, and diagrams

### 🧠 Intelligent Assistance
- Grammar, spelling, and style suggestions on demand
- Content quality assessment when requested
- Technical accuracy verification for scientific concepts

### 🔍 Contextual Research Engine
- Automated information gathering based on your writing topic
- Knowledge database that builds as you research
- Context-aware search that returns only relevant information
- Integration with academic sources and scientific databases

### 🧩 Modular Text Architecture
- Treat paragraphs as "cells" or units that can be operated on
- Apply transformations to specific content blocks
- Automatically verify content against sources
- Insert and format scientific elements (formulas, citations, etc.)

### 📤 Automated Medium Publishing
- Direct publishing to Medium after content meets quality standards
- Proper formatting of scientific elements
- Handling of complex layouts and visual elements

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ScienceMedium.git
cd ScienceMedium

# Install dependencies
pip install -r requirements.txt

# Set up configuration
python setup.py
```

## Configuration

Before using ScienceMedium, you'll need to:

1. Obtain a Medium API Integration Token
   - Log in to your Medium account
   - Go to Settings > Integration tokens
   - Create a new token and copy it

2. Configure the application
   ```bash
   python src/configure.py --medium-token "your_token_here"
   ```

3. (Optional) Configure HuggingFace API access for enhanced AI features
   ```bash
   python src/configure.py --huggingface-token "your_hf_token_here"
   ```

## Usage

### Basic Workflow

1. **Create a new article**
   ```bash
   python src/main.py new "My Scientific Article Title"
   ```

2. **Write and edit content**
   - Edit the created markdown file in your preferred editor
   - Use special syntax for research queries and AI assistance

3. **Request AI assistance**
   ```bash
   python src/main.py assist article_filename.md
   ```

4. **Perform contextual research**
   ```bash
   python src/main.py research article_filename.md "research query"
   ```

5. **Validate content**
   ```bash
   python src/main.py validate article_filename.md
   ```

6. **Publish to Medium**
   ```bash
   python src/main.py publish article_filename.md
   ```

### Special Syntax

ScienceMedium uses special syntax to trigger specific functions:

- `[?query]` - Research query that searches for information
- `[!check]` - Request grammar and style check for the paragraph
- `[#formula]` - Insert a LaTeX formula
- `[@citation]` - Insert a citation from your knowledge base
- `[^footnote]` - Create a footnote

## Project Structure

```
ScienceMedium/
├── README.md               # This documentation
├── requirements.txt        # Python dependencies
├── setup.py                # Installation script
├── src/                    # Source code
│   ├── __init__.py         # Package initialization
│   ├── main.py             # Main entry point
│   ├── configure.py        # Configuration utility
│   │   ├── __init__.py
│   │   ├── quality.py      # Content quality assessment
│   │   └── task.py         # Task management
│   ├── assistance/         # AI writing assistance
│   │   ├── __init__.py
│   │   ├── grammar.py      # Grammar checking
│   │   └── suggestions.py  # Content improvement
│   ├── research/           # Research engine
│   │   ├── __init__.py
│   │   ├── search.py       # Search functionality
│   │   └── knowledge.py    # Knowledge database
│   ├── publishing/         # Medium publishing
│   │   ├── __init__.py
│   │   ├── converter.py    # Format conversion
│   │   └── api.py          # Medium API integration
│   └── utils/              # Utility functions
│       ├── __init__.py
│       └── file.py         # File operations
└── examples/               # Example articles and configurations
    └── sample_article.md   # Sample article with annotations
```

## Requirements

- Python 3.8+
- Medium account with API access
- Internet connection for research features
- (Optional) HuggingFace account for enhanced AI features

## Contributing

This project is in active development. Contributions are welcome through:
- Bug reports and feature suggestions
- Pull requests for new features or bug fixes
- Documentation improvements

## License

This project is available under the MIT License - see the LICENSE file for details. 