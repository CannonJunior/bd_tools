#!/usr/bin/env python3
"""
Advanced Document Evaluation Client
Streamlined agent interface for superior user experience
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

console = Console()

class EvaluationStyle(Enum):
    """Available evaluation styles"""
    QUICK = "quick"
    COMPREHENSIVE = "comprehensive" 
    DETAILED = "detailed"

@dataclass
class EvaluationConfig:
    """Client configuration"""
    model_name: str = "llama3.2"
    evaluation_style: EvaluationStyle = EvaluationStyle.COMPREHENSIVE
    output_format: str = "rich"  # "rich", "json", "markdown"
    save_results: bool = False
    results_path: Optional[str] = None

class DocEvalClient:
    """Advanced document evaluation client"""
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config or EvaluationConfig()
        self.server_tools = None
        
    async def connect(self) -> bool:
        """Initialize connection to MCP server tools"""
        try:
            # Import server tools directly for this implementation
            # In production, this would use proper MCP client protocol
            from . import server
            self.server_tools = server
            
            console.print("✅ Connected to DocEval MCP Server", style="green")
            return True
            
        except Exception as e:
            console.print(f"❌ Failed to connect: {e}", style="red")
            return False
    
    async def evaluate_document(
        self,
        document_path: str,
        criteria_path: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate document with rich progress feedback
        
        Args:
            document_path: Path to document
            criteria_path: Path to criteria
            **kwargs: Override config options
            
        Returns:
            Evaluation results
        """
        # Update config with any overrides
        config = self._update_config(kwargs)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            # Step 1: Validate inputs
            task = progress.add_task("🔍 Validating inputs...", total=None)
            if not self._validate_inputs(document_path, criteria_path):
                return {"error": "Input validation failed", "success": False}
            
            # Step 2: Parse document
            progress.update(task, description="📄 Parsing document...")
            doc_result = self.server_tools.parse_document(document_path)
            if not doc_result.get("success"):
                return doc_result
            
            # Step 3: Parse criteria
            progress.update(task, description="📋 Parsing criteria...")
            criteria_result = self.server_tools.parse_criteria(criteria_path)
            if not criteria_result.get("success"):
                return criteria_result
            
            # Step 4: AI evaluation
            progress.update(task, description="🤖 Running AI evaluation...")
            eval_result = self.server_tools.evaluate_document_advanced(
                doc_result["content"],
                criteria_result,
                config.model_name,
                config.evaluation_style.value
            )
            if not eval_result.get("success"):
                return eval_result
            
            # Step 5: Advanced metrics
            progress.update(task, description="📊 Calculating metrics...")
            metrics_result = self.server_tools.calculate_advanced_metrics(eval_result)
            
            # Step 6: Compile results
            progress.update(task, description="✨ Finalizing results...")
            
        results = {
            "document_info": {
                "path": document_path,
                "word_count": doc_result["word_count"],
                "char_count": doc_result["char_count"],
                "format": doc_result["format"]
            },
            "criteria_info": {
                "path": criteria_path,
                "count": criteria_result["count"],
                "total_weight": criteria_result["total_weight"]
            },
            "evaluation": eval_result["evaluation"],
            "advanced_metrics": metrics_result,
            "config": {
                "model": config.model_name,
                "style": config.evaluation_style.value
            },
            "success": True
        }
        
        # Save results if requested
        if config.save_results:
            self._save_results(results, config.results_path)
        
        return results
    
    def _update_config(self, overrides: Dict[str, Any]) -> EvaluationConfig:
        """Update configuration with overrides"""
        config = EvaluationConfig(
            model_name=overrides.get("model_name", self.config.model_name),
            evaluation_style=EvaluationStyle(overrides.get("evaluation_style", self.config.evaluation_style.value)),
            output_format=overrides.get("output_format", self.config.output_format),
            save_results=overrides.get("save_results", self.config.save_results),
            results_path=overrides.get("results_path", self.config.results_path)
        )
        return config
    
    def _validate_inputs(self, document_path: str, criteria_path: str) -> bool:
        """Validate input files exist and are accessible"""
        doc_path = Path(document_path)
        crit_path = Path(criteria_path)
        
        if not doc_path.exists():
            console.print(f"❌ Document not found: {document_path}", style="red")
            return False
        
        if not crit_path.exists():
            console.print(f"❌ Criteria not found: {criteria_path}", style="red")
            return False
        
        return True
    
    def _save_results(self, results: Dict[str, Any], path: Optional[str] = None):
        """Save results to file"""
        if not path:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"evaluation_results_{timestamp}.json"
        
        try:
            with open(path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            console.print(f"💾 Results saved: {path}", style="green")
        except Exception as e:
            console.print(f"⚠️ Failed to save results: {e}", style="yellow")
    
    def display_results(self, results: Dict[str, Any], format_type: str = "rich"):
        """Display results in specified format"""
        if not results.get("success"):
            console.print(f"❌ Error: {results.get('error', 'Unknown error')}", style="red")
            return
        
        if format_type == "rich":
            self._display_rich_results(results)
        elif format_type == "json":
            console.print_json(data=results)
        elif format_type == "markdown":
            self._display_markdown_results(results)
        else:
            console.print("Unsupported display format", style="red")
    
    def _display_rich_results(self, results: Dict[str, Any]):
        """Display results with rich formatting"""
        console.rule("📊 Document Evaluation Results", style="blue")
        
        # Document info
        doc_info = results.get("document_info", {})
        console.print(Panel(
            f"📄 **Document:** {doc_info.get('path', 'Unknown')}\n"
            f"📝 **Format:** {doc_info.get('format', 'Unknown').upper()}\n"
            f"📊 **Words:** {doc_info.get('word_count', 0):,}\n"
            f"🔤 **Characters:** {doc_info.get('char_count', 0):,}",
            title="Document Information",
            border_style="blue"
        ))
        
        # Criteria info
        crit_info = results.get("criteria_info", {})
        console.print(Panel(
            f"📋 **Criteria File:** {crit_info.get('path', 'Unknown')}\n"
            f"🎯 **Criteria Count:** {crit_info.get('count', 0)}\n"
            f"⚖️ **Total Weight:** {crit_info.get('total_weight', 0):.1f}",
            title="Evaluation Criteria",
            border_style="green"
        ))
        
        # Scores table
        evaluation = results.get("evaluation", {})
        scores = evaluation.get("scores", [])
        
        if scores:
            table = Table(title="📈 Individual Scores")
            table.add_column("Criterion", style="cyan", width=20)
            table.add_column("Score", justify="center", style="magenta")
            table.add_column("Percentage", justify="center", style="green")
            table.add_column("Confidence", justify="center", style="yellow")
            table.add_column("Justification", style="white", width=40)
            
            for score in scores:
                percentage = f"{score.get('percentage', 0):.1f}%"
                confidence = f"{score.get('confidence', 0):.2f}"
                justification = score.get('justification', '')[:60] + "..." if len(score.get('justification', '')) > 60 else score.get('justification', '')
                
                table.add_row(
                    score.get('criterion', 'Unknown'),
                    f"{score.get('score', 0):.1f}/{score.get('max_score', 10)}",
                    percentage,
                    confidence,
                    justification
                )
            
            console.print(table)
        
        # Overall results
        console.print(Panel(
            f"🏆 **Overall Score:** {evaluation.get('overall_score', 0):.1f}/10\n"
            f"⚖️ **Weighted Score:** {evaluation.get('weighted_score', 0):.1f}\n"
            f"📊 **Percentage:** {evaluation.get('percentage', 0):.1f}%\n\n"
            f"📝 **Summary:**\n{evaluation.get('summary', 'No summary available')}",
            title="Overall Assessment",
            border_style="yellow"
        ))
        
        # Recommendations
        recommendations = evaluation.get('recommendations', [])
        if recommendations:
            rec_text = "\n".join(f"• {rec}" for rec in recommendations)
            console.print(Panel(
                rec_text,
                title="💡 Recommendations",
                border_style="cyan"
            ))
        
        # Advanced metrics
        metrics = results.get("advanced_metrics", {})
        if metrics and metrics.get("success"):
            self._display_advanced_metrics(metrics)
    
    def _display_advanced_metrics(self, metrics: Dict[str, Any]):
        """Display advanced metrics panel"""
        stats = metrics.get("statistical_analysis", {})
        bands = metrics.get("performance_bands", {})
        risk = metrics.get("risk_assessment", {})
        
        # Statistical analysis
        stats_text = (
            f"📊 **Mean Score:** {stats.get('mean', 0):.2f}\n"
            f"📈 **Std Dev:** {stats.get('std_dev', 0):.2f}\n"
            f"🎯 **Consistency:** {metrics.get('consistency_score', 0):.2f}\n"
            f"📉 **Range:** {stats.get('range', 0):.1f}"
        )
        
        # Performance bands
        excellent = len(bands.get('excellent', []))
        good = len(bands.get('good', []))
        fair = len(bands.get('fair', []))
        poor = len(bands.get('poor', []))
        
        bands_text = (
            f"🌟 **Excellent:** {excellent}\n"
            f"✅ **Good:** {good}\n"
            f"⚠️ **Fair:** {fair}\n"
            f"❌ **Poor:** {poor}"
        )
        
        # Risk assessment
        risk_text = (
            f"🚨 **Risk Level:** {risk.get('risk_level', 'unknown').title()}\n"
            f"⚠️ **Critical Issues:** {risk.get('critical_issues', 0)}\n"
            f"🔍 **Concerning Areas:** {risk.get('concerning_areas', 0)}"
        )
        
        console.print(Panel(
            f"{stats_text}\n\n{bands_text}\n\n{risk_text}",
            title="📊 Advanced Metrics",
            border_style="magenta"
        ))
    
    def _display_markdown_results(self, results: Dict[str, Any]):
        """Display results as markdown"""
        evaluation = results.get("evaluation", {})
        
        markdown_content = f"""
# Document Evaluation Results

## Document Information
- **Path:** {results.get('document_info', {}).get('path', 'Unknown')}
- **Format:** {results.get('document_info', {}).get('format', 'Unknown')}
- **Word Count:** {results.get('document_info', {}).get('word_count', 0):,}

## Overall Assessment
- **Score:** {evaluation.get('overall_score', 0):.1f}/10
- **Percentage:** {evaluation.get('percentage', 0):.1f}%

## Summary
{evaluation.get('summary', 'No summary available')}

## Individual Scores
"""
        
        for score in evaluation.get('scores', []):
            markdown_content += f"""
### {score.get('criterion', 'Unknown')}
- **Score:** {score.get('score', 0):.1f}/{score.get('max_score', 10)}
- **Percentage:** {score.get('percentage', 0):.1f}%
- **Justification:** {score.get('justification', 'No justification')}
"""
        
        if evaluation.get('recommendations'):
            markdown_content += "\n## Recommendations\n"
            for rec in evaluation.get('recommendations', []):
                markdown_content += f"- {rec}\n"
        
        console.print(Markdown(markdown_content))
    
    async def get_available_models(self) -> List[str]:
        """Get list of available Ollama models"""
        try:
            import ollama
            models = ollama.list()
            return [model["name"] for model in models.get("models", [])]
        except Exception as e:
            logger.warning(f"Could not get Ollama models: {e}")
            return ["llama3.2", "llama3.1", "llama2"]
    
    async def quick_evaluate(self, document_path: str, criteria_path: str) -> Dict[str, Any]:
        """Quick evaluation with minimal output"""
        results = await self.evaluate_document(
            document_path, 
            criteria_path,
            evaluation_style="quick",
            output_format="rich"
        )
        
        if results.get("success"):
            evaluation = results.get("evaluation", {})
            console.print(f"✅ Quick Evaluation Complete")
            console.print(f"📊 Score: {evaluation.get('overall_score', 0):.1f}/10 ({evaluation.get('percentage', 0):.1f}%)")
        
        return results

async def main():
    """Example usage"""
    client = DocEvalClient()
    
    if await client.connect():
        console.print("🚀 DocEval Client Ready!")
        
        # Show available models
        models = await client.get_available_models()
        console.print(f"📱 Available models: {', '.join(models[:3])}")
        
        console.print("\n💡 Example usage:")
        console.print("client = DocEvalClient()")
        console.print("await client.connect()")
        console.print("results = await client.evaluate_document('doc.pdf', 'criteria.json')")
        console.print("client.display_results(results)")

if __name__ == "__main__":
    asyncio.run(main())