import asyncio
import json
import requests
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3.5:4b"


def call_ollama(messages, tools=None):
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "think": False,
    }

    if tools:
        payload["tools"] = tools

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling Ollama: {e}")
        return None


def convert_mcp_tool_to_ollama_format(mcp_tool):
    return {
        "type": "function",
        "function": {
            "name": mcp_tool.name,
            "description": mcp_tool.description,
            "parameters": mcp_tool.inputSchema
        }
    }


async def run_mcp_demo_with_ollama():
    import os
    import sys

    python_path = sys.executable
    server_script = os.path.join(os.path.dirname(__file__), "mcp_server.py")

    server_params = StdioServerParameters(
        command=python_path,
        args=[server_script],
        env=None
    )

    print("Starting MCP Server...")
    print("=" * 60)

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools_list = await session.list_tools()
            print(f"Connected to MCP server with {len(tools_list.tools)} tools:")
            for tool in tools_list.tools:
                print(f"   - {tool.name}: {tool.description}")
            print()

            ollama_tools = [
                convert_mcp_tool_to_ollama_format(tool)
                for tool in tools_list.tools
            ]

            questions = [
                "What is the weather in the capital of Germany?",
                "What is the population of France and what's the weather like in Paris?",
                "Tell me about Japan - what's its capital and what's the weather there?"
            ]

            for i, question in enumerate(questions, 1):
                print(f"\n{'=' * 60}")
                print(f"Question {i}: {question}")
                print('=' * 60)

                messages = [{"role": "user", "content": question}]

                # Agentic loop: allow the model to chain tool calls
                max_iterations = 10
                iteration = 0

                while iteration < max_iterations:
                    iteration += 1

                    response_data = call_ollama(messages, tools=ollama_tools)

                    if not response_data:
                        print("Failed to get response from Ollama")
                        break

                    assistant_message = response_data.get("message", {})

                    tool_calls = assistant_message.get("tool_calls", [])

                    if tool_calls:
                        messages.append(assistant_message)

                        for tool_call in tool_calls:
                            function_name = tool_call["function"]["name"]
                            function_args = tool_call["function"]["arguments"]

                            print(f"\nCalling tool: {function_name}")
                            print(f"   Input: {json.dumps(function_args, indent=2)}")

                            try:
                                result = await session.call_tool(function_name, function_args)

                                result_text = ""
                                for content in result.content:
                                    if hasattr(content, "text"):
                                        result_text += content.text

                                print(f"   Result: {result_text[:200]}...")

                                messages.append({
                                    "role": "tool",
                                    "content": result_text
                                })

                            except Exception as e:
                                print(f"   Error calling tool: {e}")
                                messages.append({
                                    "role": "tool",
                                    "content": f"Error: {str(e)}"
                                })
                    else:
                        content = assistant_message.get("content", "")
                        if content:
                            print(f"\nAnswer:\n{content}")
                        break

                if iteration >= max_iterations:
                    print(f"\nReached maximum iterations ({max_iterations})")

            print(f"\n\n{'=' * 60}")
            print("Demo completed successfully!")
            print('=' * 60)


async def check_ollama_availability():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        response.raise_for_status()

        models_data = response.json()
        available_models = [m["name"] for m in models_data.get("models", [])]

        if not any(MODEL in m for m in available_models):
            print(f"Warning: Model '{MODEL}' not found in Ollama")
            print(f"Available models: {', '.join(available_models)}")
            print(f"\nTo pull the model, run: ollama pull {MODEL}")
            return False

        return True

    except requests.exceptions.RequestException as e:
        print(f"Error: Cannot connect to Ollama at http://localhost:11434")
        print(f"Make sure Ollama is running with: ollama serve")
        print(f"Error details: {e}")
        return False


if __name__ == "__main__":
    print("MCP Client with Ollama - Open Source LLM Demo")
    print(f"Model: {MODEL}")
    print()

    if not asyncio.run(check_ollama_availability()):
        print("\nCannot proceed without Ollama running")
        exit(1)

    try:
        asyncio.run(run_mcp_demo_with_ollama())
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
