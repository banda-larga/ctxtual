"""
Framework integration adapters for ctx.

Each sub-module provides zero-hard-dependency adapters for a specific
framework.  Import the one you need::

    from ctxtual.integrations.openai import handle_tool_calls
    from ctxtual.integrations.anthropic import handle_tool_use
    from ctxtual.integrations.langchain import to_langchain_tools

No additional dependencies are required for the OpenAI and Anthropic
adapters — they work with both SDK response objects and raw dicts.
The LangChain adapter requires ``langchain-core``.
"""
