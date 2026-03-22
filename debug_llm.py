import sys
sys.path.append("/app")
import asyncio
import logging
import os
import anthropic
from executor import SYSTEM_PROMPT, ANTHROPIC_AVAILABLE, dispatch_tool

logging.basicConfig(level=logging.INFO)

tools = [
    {
        "name": "search_zip_corpus",
        "description": "A perfectly general RAG tool to dynamically fetch and search a compressed corpus of files. Downloads a remote ZIP file containing text, json, or csv files into memory. You can optionally filter which files to search using filename_filter. Returns ONLY the paragraphs or lines matching your keyword to save tokens.",
        "input_schema": {
            "type": "object",
            "properties": {
                "zip_url": {"type": "string", "description": "The exact URL of the .zip file containing the corpus."},
                "filename_filter": {"type": "string", "description": "Optional substring to filter files (e.g., '1940' or '1953_01')."},
                "keyword": {"type": "string", "description": "Keyword to extract relevant paragraphs from the files. Case-insensitive."}
            },
            "required": ["zip_url"]
        }
    },
    {
        "name": "execute_python",
        "description": "Execute Python code for mathematical calculations. Use for regressions, standard deviations, geometric means, inflation adjustments, and any numeric computation. pandas and numpy are available. Print your final result.",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute. Always print() your final answer."}
            },
            "required": ["code"]
        }
    }
]

async def test_question(prompt: str):
    client = anthropic.Anthropic()
    model = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
    
    messages = [{"role": "user", "content": prompt}]
    
    print(f"\n--- TESTING: {prompt} ---\n")
    
    for step in range(15):
        print(f"\n--- STEP {step+1} ---")
        try:
            response = client.messages.create(
                model=model, max_tokens=4096, system=SYSTEM_PROMPT,
                messages=messages, tools=tools, temperature=0,
            )
            
            for block in response.content:
                if block.type == "text":
                    print(f"CLAUDE:\n{block.text}\n")
                elif block.type == "tool_use":
                    print(f"TOOL CALL: {block.name}({block.input})")
                    result = dispatch_tool(block.name, block.input)
                    print(f"RESULT: {str(result)[:200]}...\n")
                    
                    messages.append({"role": "assistant", "content": response.content})
                    messages.append({
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": block.id, "content": result}]
                    })
            if response.stop_reason == "end_turn":
                print("--- FINISHED ---")
                break
        except Exception as e:
            print(f"ERROR: {e}")
            break

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        asyncio.run(test_question(sys.argv[1]))
    else:
        print("Please provide a question.")
