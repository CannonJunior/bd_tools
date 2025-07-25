# =� DocEval - Advanced Document Evaluation System

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastMCP](https://img.shields.io/badge/FastMCP-latest-green.svg)](https://github.com/anthropic/fastmcp)
[![Ollama](https://img.shields.io/badge/Ollama-compatible-orange.svg)](https://ollama.ai/)

> **Superior AI-powered document evaluation with MCP integration**
> 
> Built to outperform existing solutions with advanced features, intuitive interfaces, and comprehensive analytics.

## ( Features

### <� **Superior Capabilities**
- **=� Multi-Format Support**: PDF, DOC, DOCX, TXT, RTF, HTML, MD
- **=� Flexible Criteria**: JSON, CSV, XLSX, YAML with weighted scoring
- **> AI-Powered Analysis**: Advanced prompting with confidence scoring
- **=� Rich Analytics**: Statistical analysis, risk assessment, recommendations
- **<� Beautiful UI**: Rich terminal output with progress indicators

### =' **Technical Excellence**
- **� FastMCP Integration**: Standard MCP protocol with async tools
- **<� Modular Architecture**: Extensible, maintainable codebase
- **=� Advanced Metrics**: Performance bands, consistency scoring, priorities
- **=� Robust Error Handling**: Graceful failures with detailed feedback
- **=� High Performance**: Parallel processing, caching, optimization

### <� **User Experience**
- **=� Powerful CLI**: Rich terminal interface with Typer
- **= Python API**: Async client for integration
- **=� Multiple Output Formats**: Rich, JSON, Markdown
- **=� Export Capabilities**: Save results in multiple formats

## =� Quick Start

### Installation

```bash
# Clone and install
cd doc_eval
uv pip install .

# Install Ollama (required for AI evaluation)
# Visit https://ollama.ai for installation instructions
ollama pull llama3.2
```

### Basic Usage

```bash
# Validate files first
doc-eval validate document.pdf criteria.json

# Run comprehensive evaluation
doc-eval evaluate document.pdf criteria.json

# Quick assessment
doc-eval quick document.txt criteria.csv

# Custom model and style
doc-eval evaluate document.docx criteria.xlsx --model llama3.1 --style detailed

# Save results
doc-eval evaluate document.pdf criteria.json --save results.json
```

### Python API

```python
import asyncio
from doc_eval.client import DocEvalClient

async def main():
    client = DocEvalClient()
    await client.connect()
    
    results = await client.evaluate_document(
        "document.pdf", 
        "criteria.json"
    )
    
    client.display_results(results)

asyncio.run(main())
```

## =� Criteria Formats

### JSON Format (Recommended)
```json
{
  "criteria": [
    {
      "name": "Technical Accuracy",
      "description": "Factual correctness and precision",
      "weight": 3.0,
      "max_score": 10,
      "category": "Content Quality"
    }
  ]
}
```

### CSV Format
```csv
name,description,weight,max_score,category
Content Quality,Overall quality and accuracy,3.0,10,Core
Readability,Ease of reading and understanding,2.5,10,Core
```

## <� Architecture

```
                                                           
   Rich CLI             Python API           MCP Client    
   (Typer)              (Async)              (FastMCP)     
                                                           
                                                       
                                <                       
                                 
                                     
                       MCP Server    
                       (FastMCP)     
                                     
                             
                                     
                         Ollama      
                       (Local LLM)   
                                     
```

## =� Available Commands

### Core Commands
- `doc-eval evaluate` - Comprehensive document evaluation
- `doc-eval quick` - Fast assessment with minimal output
- `doc-eval validate` - Verify files without evaluation
- `doc-eval server` - Start MCP server

### Utility Commands
- `doc-eval models` - List available Ollama models
- `doc-eval info` - System information and capabilities
- `doc-eval examples` - Usage examples and samples

## =� Output Examples

### Rich Terminal Output
```
=� Document Evaluation Results
PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP

  Document Information  
 =� Document: report.pdf 
 =� Format: PDF          
 =� Words: 1,234         
 =$ Characters: 6,789    
                         

=� Individual Scores
                  ,        ,            ,            ,                 
 Criterion         Score   Percentage  Confidence  Justification   
                  <        <            <            <                 $
 Technical Acc...  8.5/10  85.0%       0.92        Well researched 
 Clarity           7.2/10  72.0%       0.88        Generally clear 
                  4        4            4            4                 
```

## >� Testing

Run the validation test:
```bash
doc-eval validate examples/sample_document.txt examples/comprehensive_criteria.json
```

## =� Advanced Features

### Evaluation Styles
- **Quick**: Fast assessment for rapid feedback
- **Comprehensive**: Balanced analysis with good detail
- **Detailed**: In-depth evaluation with extensive recommendations

### Advanced Metrics
- Statistical analysis (mean, std dev, consistency)
- Performance band categorization (excellent/good/fair/poor)
- Risk assessment and priority identification
- Improvement recommendations and action items

### Integration Options
- **MCP Server**: Standard protocol for AI agent integration
- **Python API**: Async/await support for modern applications  
- **CLI Tool**: Rich terminal interface for interactive use
- **Export Formats**: JSON, Markdown, Rich console output

## =� Examples

See the `examples/` directory for:
- `sample_document.txt` - Comprehensive test document
- `comprehensive_criteria.json` - Full-featured criteria example
- `simple_criteria.csv` - Basic CSV criteria format

## <� Competitive Advantages

### vs. GEMINI's document_evaluator:
-  **Superior Architecture**: Clean MCP integration vs mixed approaches
-  **Rich UI**: Beautiful terminal output vs plain text
-  **Advanced Features**: Statistical metrics, confidence scoring, risk assessment
-  **Better UX**: Intuitive CLI with validation, examples, help
-  **Robust Error Handling**: Graceful failures with detailed feedback
-  **Performance**: Async processing, caching, optimization

### Key Differentiators:
1. **Professional CLI** with rich terminal output and progress indicators
2. **Advanced Analytics** beyond basic scoring (statistics, risk, recommendations)
3. **Multiple Evaluation Styles** for different use cases
4. **Comprehensive Validation** with detailed error reporting
5. **Export Flexibility** with multiple output formats
6. **Better Architecture** with clean separation of concerns

## =� License

Built for the bd_tools project. Advanced AI-powered document evaluation system.

---

**=� DocEval - Where AI meets Document Excellence**