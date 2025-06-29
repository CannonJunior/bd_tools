#!/usr/bin/env python3
"""
DocEval CLI - Advanced command line interface
Superior user experience with rich output and comprehensive options
"""

import asyncio
from pathlib import Path
from typing import Optional, List
from enum import Enum

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from .client import DocEvalClient, EvaluationConfig, EvaluationStyle

# Initialize Typer app and Rich console
app = typer.Typer(
    name="doc-eval",
    help="🚀 Advanced AI-powered document evaluation system",
    epilog="Built with FastMCP and Ollama integration",
    rich_markup_mode="rich"
)
console = Console()

class OutputFormat(str, Enum):
    """Available output formats"""
    RICH = "rich"
    JSON = "json"
    MARKDOWN = "markdown"

@app.command()
def evaluate(
    document: str = typer.Argument(..., help="📄 Path to document file"),
    criteria: str = typer.Argument(..., help="📋 Path to criteria file"),
    model: str = typer.Option("llama3.2", "--model", "-m", help="🤖 Ollama model to use"),
    style: EvaluationStyle = typer.Option(
        EvaluationStyle.COMPREHENSIVE, 
        "--style", "-s", 
        help="🎯 Evaluation style"
    ),
    output_format: OutputFormat = typer.Option(
        OutputFormat.RICH, 
        "--format", "-f", 
        help="📊 Output format"
    ),
    save: Optional[str] = typer.Option(None, "--save", help="💾 Save results to file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="🔍 Verbose output"),
):
    """
    🚀 Evaluate a document against criteria using AI analysis
    
    This command performs comprehensive document evaluation with:
    - Advanced document parsing (PDF, DOC, DOCX, TXT, etc.)
    - Flexible criteria formats (JSON, CSV, XLSX)  
    - AI-powered scoring with justifications
    - Advanced metrics and recommendations
    """
    
    asyncio.run(_run_evaluation(
        document, criteria, model, style, output_format, save, verbose
    ))

async def _run_evaluation(
    document: str,
    criteria: str, 
    model: str,
    style: EvaluationStyle,
    output_format: OutputFormat,
    save: Optional[str],
    verbose: bool
):
    """Internal evaluation runner"""
    
    # Create client configuration
    config = EvaluationConfig(
        model_name=model,
        evaluation_style=style,
        output_format=output_format.value,
        save_results=save is not None,
        results_path=save
    )
    
    client = DocEvalClient(config)
    
    # Show header
    console.rule("🚀 DocEval - Document Evaluation System", style="blue bold")
    
    if verbose:
        console.print(f"📄 Document: {document}")
        console.print(f"📋 Criteria: {criteria}")
        console.print(f"🤖 Model: {model}")
        console.print(f"🎯 Style: {style.value}")
        console.print()
    
    # Connect to server
    if not await client.connect():
        raise typer.Exit(1)
    
    # Run evaluation
    results = await client.evaluate_document(document, criteria)
    
    # Display results
    client.display_results(results, output_format.value)
    
    if results.get("success"):
        console.print("\n✅ Evaluation completed successfully!", style="green bold")
    else:
        console.print(f"\n❌ Evaluation failed: {results.get('error', 'Unknown error')}", style="red bold")
        raise typer.Exit(1)

@app.command()
def quick(
    document: str = typer.Argument(..., help="📄 Path to document file"),
    criteria: str = typer.Argument(..., help="📋 Path to criteria file"),
    model: str = typer.Option("llama3.2", "--model", "-m", help="🤖 Ollama model to use"),
):
    """
    ⚡ Quick document evaluation with minimal output
    
    Performs a streamlined evaluation focused on core metrics.
    Perfect for rapid assessments and batch processing.
    """
    
    asyncio.run(_run_quick_evaluation(document, criteria, model))

async def _run_quick_evaluation(document: str, criteria: str, model: str):
    """Internal quick evaluation runner"""
    
    config = EvaluationConfig(
        model_name=model,
        evaluation_style=EvaluationStyle.QUICK,
        output_format="rich"
    )
    
    client = DocEvalClient(config)
    
    if not await client.connect():
        raise typer.Exit(1)
    
    await client.quick_evaluate(document, criteria)

@app.command()
def models():
    """
    📱 List available Ollama models
    
    Shows all locally available language models that can be used
    for document evaluation.
    """
    
    asyncio.run(_list_models())

async def _list_models():
    """List available models"""
    
    client = DocEvalClient()
    
    if not await client.connect():
        raise typer.Exit(1)
    
    models = await client.get_available_models()
    
    console.rule("📱 Available Ollama Models", style="cyan")
    
    if models:
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Model Name", style="green")
        table.add_column("Recommended For", style="yellow")
        
        model_recommendations = {
            "llama3.2": "General purpose, balanced performance",
            "llama3.1": "High accuracy, detailed analysis", 
            "llama2": "Lightweight, faster processing",
            "mistral": "Technical documents, coding",
            "codellama": "Code documentation, technical content"
        }
        
        for model in models:
            recommendation = model_recommendations.get(model, "General use")
            table.add_row(model, recommendation)
        
        console.print(table)
    else:
        console.print("❌ No Ollama models found. Please install Ollama and pull some models.", style="red")

@app.command()
def server():
    """
    🖥️ Start the DocEval MCP server
    
    Launches the Model Context Protocol server that provides
    document evaluation tools for MCP clients.
    """
    
    console.rule("🖥️ Starting DocEval MCP Server", style="green")
    console.print("🚀 Server starting...")
    console.print("📡 MCP tools available:")
    console.print("  • parse_document")
    console.print("  • parse_criteria") 
    console.print("  • evaluate_document_advanced")
    console.print("  • calculate_advanced_metrics")
    console.print("  • full_document_evaluation")
    console.print()
    console.print("Press Ctrl+C to stop", style="yellow")
    
    try:
        from .server import main as server_main
        server_main()
    except KeyboardInterrupt:
        console.print("\n🛑 Server stopped", style="yellow")

@app.command()
def validate(
    document: str = typer.Argument(..., help="📄 Path to document file"),
    criteria: str = typer.Argument(..., help="📋 Path to criteria file"),
):
    """
    ✅ Validate document and criteria files
    
    Checks that files exist, are readable, and have supported formats
    without running a full evaluation.
    """
    
    asyncio.run(_validate_files(document, criteria))

async def _validate_files(document: str, criteria: str):
    """Validate input files"""
    
    console.rule("✅ File Validation", style="yellow")
    
    client = DocEvalClient()
    
    if not await client.connect():
        raise typer.Exit(1)
    
    # Check document
    doc_path = Path(document)
    if not doc_path.exists():
        console.print(f"❌ Document not found: {document}", style="red")
        raise typer.Exit(1)
    
    # Check criteria  
    crit_path = Path(criteria)
    if not crit_path.exists():
        console.print(f"❌ Criteria not found: {criteria}", style="red")
        raise typer.Exit(1)
    
    # Test parsing
    try:
        from .server import parse_document, parse_criteria
        
        console.print("🔍 Testing document parsing...")
        doc_result = parse_document(document)
        if doc_result.get("success"):
            console.print(f"✅ Document parsed successfully ({doc_result['word_count']} words)", style="green")
        else:
            console.print(f"❌ Document parsing failed: {doc_result.get('error')}", style="red")
            raise typer.Exit(1)
        
        console.print("🔍 Testing criteria parsing...")
        crit_result = parse_criteria(criteria)
        if crit_result.get("success"):
            console.print(f"✅ Criteria parsed successfully ({crit_result['count']} criteria)", style="green")
        else:
            console.print(f"❌ Criteria parsing failed: {crit_result.get('error')}", style="red")
            raise typer.Exit(1)
        
        console.print("\n🎉 All validations passed!", style="green bold")
        
    except Exception as e:
        console.print(f"❌ Validation error: {e}", style="red")
        raise typer.Exit(1)

@app.command()
def info():
    """
    ℹ️ Show system information and capabilities
    
    Displays information about DocEval capabilities, supported formats,
    and system status.
    """
    
    console.rule("ℹ️ DocEval System Information", style="blue")
    
    # System info
    console.print(Panel(
        "🚀 **DocEval v0.1.0**\n"
        "Advanced AI-powered document evaluation system\n"
        "Built with FastMCP, Ollama, and Rich UI",
        title="About",
        border_style="blue"
    ))
    
    # Supported formats
    console.print(Panel(
        "📄 **Documents:** PDF, DOC, DOCX, TXT, RTF, HTML, MD\n"
        "📋 **Criteria:** JSON, CSV, XLSX, YAML\n"
        "🤖 **Models:** Any Ollama-compatible LLM\n"
        "📊 **Output:** Rich console, JSON, Markdown",
        title="Supported Formats",
        border_style="green"
    ))
    
    # Features
    console.print(Panel(
        "🎯 **Advanced Scoring:** 1-10 scale with justifications\n"
        "⚖️ **Weighted Assessment:** Customizable criterion weights\n"
        "📊 **Rich Metrics:** Statistical analysis and insights\n"
        "💡 **Recommendations:** AI-powered improvement suggestions\n"
        "🚀 **MCP Integration:** Standard protocol compatibility",
        title="Key Features", 
        border_style="yellow"
    ))

@app.command() 
def examples():
    """
    📚 Show usage examples and sample commands
    
    Provides comprehensive examples of how to use DocEval
    for different evaluation scenarios.
    """
    
    console.rule("📚 DocEval Usage Examples", style="cyan")
    
    examples_text = """
**Basic Evaluation:**
```bash
doc-eval evaluate document.pdf criteria.json
```

**Quick Assessment:**
```bash
doc-eval quick report.docx requirements.csv
```

**Custom Model and Style:**
```bash
doc-eval evaluate document.txt criteria.xlsx --model llama3.1 --style detailed
```

**Save Results:**
```bash
doc-eval evaluate document.pdf criteria.json --save results.json
```

**JSON Output:**
```bash
doc-eval evaluate document.docx criteria.csv --format json
```

**Validation Only:**
```bash
doc-eval validate document.pdf criteria.json
```

**Server Mode:**
```bash
doc-eval server
```
"""
    
    from rich.markdown import Markdown
    console.print(Markdown(examples_text))

def main():
    """Entry point for the CLI application"""
    app()

if __name__ == "__main__":
    main()