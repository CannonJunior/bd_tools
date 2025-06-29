## Product Vision

This repository will create tools for Agentic AIs to use and also people.

# Rules
* 1. Use Python as a default. If there is a reason to develop a tool entirely or in part in another language, then stop to think step-by-step about the long-term utility of the tool. It may be that other tools previously developed should be refactored in this or other languages.
* 2. For Python, always use the uv package management project, documented here:
https://docs.astral.sh/uv/guides/install-python/
* 3. For tools, write them as Model Context Protocol compatible, documented here:
https://docs.anthropic.com/en/docs/mcp
Do not create a tool without also creating an MCP Server for the tool.
* 4. For agent interactions, write them consistent with the Agent2Agent protocol documented here:
https://github.com/a2aproject/a2a-python
* 5. When creating or working with agents, by default create a top level agent. This agent's only task is to delegate tasks to other agents while also checking credentials and providing them access to tools. If you plan on giving this top level agent tasks other than delegation, wait. Then stop and think step-by-step whether there is a better method than over-tasking the top level agent.
* 6. Whenever you create a new directory, create a CLAUDE.md file in that new directory. The purpose of CLAUDE.md includes
- Defining System Prompts: Users can write instructions or guidelines for the Claude Code agent within the CLAUDE.md file. This allows users to customize the AI's responses.
- Providing Context: Developers can use CLAUDE.md to define project conditions, best practices, or point the AI at codebases relevant to their workflow.
- Personal and Team Configurations: This file allows for both personal and team-level configurations, ensuring project standards.
- Enhancing AI Productivity: By providing context, CLAUDE.md helps improve the accuracy and relevance of the AI's responses, boosting developer productivity.
* 7. When creating agents and MCP tools, always make them as simple as possible, but no simpler.
* 8. With MCP tools, run them locally using the Python FastMCP library as a default. Ask the user for permission to consider other options.
