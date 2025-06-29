# Document Evaluator MCP Server

## Purpose
FastMCP-based document evaluation tools that:
- Parse documents (txt, pdf, doc formats)
- Parse evaluation criteria (CSV, JSON)
- Output numeric scores for each criterion
- Provide overall assessment

## Architecture
- MCP Server: FastMCP with evaluation tools
- MCP Client: Agent that connects to Ollama models
- Tools: Document parsing, criteria parsing, scoring, assessment

## Requirements
- Use FastMCP for both server and client
- Connect to locally running Ollama models
- Support numeric scoring (1-10 scale)
- Handle multiple document and criteria formats
- Structured JSON output with scores and justifications

## Development Notes
- Keep tools simple but comprehensive
- Use uv for package management
- Follow MCP protocol standards
- Ensure proper error handling