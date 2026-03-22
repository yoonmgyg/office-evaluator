import asyncio
import logging
import os
import re
import json
from uuid import uuid4
from functools import lru_cache

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import (
    Message,
    Part,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
    UnsupportedOperationError,
)

logger = logging.getLogger(__name__)

TERMINAL_STATES = {
    TaskState.completed,
    TaskState.canceled,
    TaskState.failed,
    TaskState.rejected,
}

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

GITHUB_TREE_URL = (
    "https://api.github.com/repos/databricks/officeqa/git/trees/6aa8c32?recursive=1"
)
RAW_BASE = (
    "https://raw.githubusercontent.com/databricks/officeqa/6aa8c32/"
)

SYSTEM_PROMPT = """You are an advanced data synthesis agent designed to solve complex analytical questions. You must ensure absolute numerical accuracy.

You have access to three tools:

1. **search_zip_corpus** — A generic RAG tool to search zipped document repositories. 
   Use this FIRST to retrieve relevant text from the target corpus. 
   For this specific scenario, your primary historical document corpus is located at:
   `https://raw.githubusercontent.com/databricks/officeqa/6aa8c32/treasury_bulletins_parsed/transformed/treasury_bulletins_transformed.zip`
   Pass this URL to the tool. Supply a `filename_filter` (like "1940" or "1953_01").
   CRITICAL: Use ONLY short, single-word keywords (like "defense" or "veterans") because the tool filters paragraphs that contain all your words. Long queries will fail!

2. **execute_python** — Use this for ALL math: regressions, standard deviations, geometric means, inflation adjustments, etc.
   Write complete Python scripts. Use pandas and numpy as needed — they are pre-installed.
   You can pass the data you received from your corpus search into your Python code.
   CRITICAL: When summing tabular data, BEWARE OF DOUBLE COUNTING! Tables often include both a "Total" row and sub-category rows. Do not sum a "Total" row with its sub-components!
   NOTE: If you need inflation data (like historical CPI-U), you MUST write a python script using `requests` and `bs4` to scrape it, as the web_search tool is disabled!

3. **web_search** — Use this as a LAST RESORT for external data not found in the corpus.

WORKFLOW:
1. Analyze the question to identify the year(s), month(s), and category/keyword.
2. Call search_zip_corpus to retrieve the relevant datasets natively within the context.
3. If the data needs math, call execute_python.
4. Output your final answer.

When you are completely finished, output your final answer as:
<REASONING>
[Brief summary of steps and calculations]
</REASONING>
<FINAL_ANSWER>
[value]
</FINAL_ANSWER>

CRITICAL FORMATTING RULES:
- The <FINAL_ANSWER> tag MUST contain ONLY the final value. No words, no explanations, no hedging.
- If the question asks for a number: just the number (e.g. 2602)
- If it asks for a percent: include the % sign (e.g. 12.34%)
- If it asks for a list in brackets: format exactly as requested (e.g. [0.096, -184.143])
- NEVER write "I cannot determine" or "Unable to find" — always make your best attempt with available data.
- ANY conversational text inside <FINAL_ANSWER> will cause AUTO-FAILURE.
"""

_corpus_caches = {}

def _get_corpus_cache(zip_url: str):
    global _corpus_caches
    if zip_url in _corpus_caches:
        return _corpus_caches[zip_url]
    
    import requests, zipfile, io
    try:
        logger.info(f"Downloading corpus zip from {zip_url}...")
        resp = requests.get(zip_url, timeout=60)
        resp.raise_for_status()
        
        cache = {}
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            for fname in z.namelist():
                if fname.endswith(".txt") or fname.endswith(".json") or fname.endswith(".csv"):
                    cache[fname] = z.read(fname).decode("utf-8", errors="replace")
        
        logger.info(f"Loaded {len(cache)} documents into memory from {zip_url}.")
        _corpus_caches[zip_url] = cache
        return cache
    except Exception as e:
        logger.error(f"Failed to fetch corpus zip: {e}")
        return {}

def _find_matching_filenames(filter_str: str, filenames: list) -> list:
    filter_str = filter_str.strip().lower()
    if not filter_str:
        return filenames
    
    matches = [fname for fname in filenames if filter_str in fname.lower()]
    return sorted(matches)

def search_zip_corpus(zip_url: str, filename_filter: str = "", keyword: str = "") -> str:
    corpus = _get_corpus_cache(zip_url)
    if not corpus:
        return f"Error: Could not retrieve or extract corpus from {zip_url}."
    
    matching_files = _find_matching_filenames(filename_filter, list(corpus.keys()))
    
    if not matching_files:
        return f"No documents found for filter='{filename_filter}'. Available files: {list(corpus.keys())[:5]}..."
    
    results = []
    keyword_words = keyword.strip().lower().split() if keyword else []
    
    for fname in matching_files[:12]:
        content = corpus[fname]
        if not keyword_words:
            results.append(f"--- {fname} ---\n{content[:4000]}... [Truncated. Provide a keyword to filter.]")
            continue
            
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        matching_paras = [p for p in paragraphs if all(w in p.lower() for w in keyword_words)]
        
        if matching_paras:
            results.append(f"--- {fname} (filtered for '{keyword}') ---")
            for p in matching_paras[:50]:
                results.append(p)
        else:
            lines = [line.strip() for line in content.split("\n") if line.strip()]
            matching_lines = [line for line in lines if all(w in line.lower() for w in keyword_words)]
            if matching_lines:
                results.append(f"--- {fname} (filtered lines for '{keyword}') ---")
                results.extend(matching_lines[:300])
    
    if not results:
        found_names = ", ".join(matching_files[:5])
        return f"Found documents ({found_names}) but none contained the keyword '{keyword}'."
        
    output = "\n".join(results)
    if len(output) > 80000:
        return output[:80000] + "\n... [Output truncated to 80000 chars]"
    return output


def execute_python_code(code: str) -> str:
    import sys, io, ast, traceback
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        tree = ast.parse(code)
        if not tree.body:
            return "No code provided"
        last = tree.body[-1]
        g = {}
        if isinstance(last, ast.Expr):
            if len(tree.body) > 1:
                exec(compile(ast.Module(body=tree.body[:-1], type_ignores=[]), "<ast>", "exec"), g)
            res_eval = eval(compile(ast.Expression(body=last.value), "<ast>", "eval"), g)
            if res_eval is not None:
                print(res_eval)
        else:
            exec(compile(tree, "<ast>", "exec"), g)
        return sys.stdout.getvalue() or "Code executed successfully."
    except Exception as e:
        return traceback.format_exc()
    finally:
        sys.stdout = old_stdout


def web_search_local(query: str) -> str:
    logger.warning(f"web_search called locally with query: {query}")
    return "Web search proxy unavailable in local testing. Use query_treasury_corpus or execute_python instead."


def extract_final_answer(response_text: str) -> str:
    match = re.search(r"<FINAL_ANSWER>(.*?)</FINAL_ANSWER>", response_text, re.DOTALL)
    if match:
        answer = match.group(1).strip()
        prefixes_to_strip = [
            "The answer is ", "The answer is: ", "Answer: ",
            "The value is ", "The result is ", "Result: ",
        ]
        for prefix in prefixes_to_strip:
            if answer.lower().startswith(prefix.lower()):
                answer = answer[len(prefix):].strip()
        return f"<FINAL_ANSWER>\n{answer}\n</FINAL_ANSWER>"
    
    if len(response_text.split()) < 20:
        return f"<FINAL_ANSWER>\n{response_text.strip()}\n</FINAL_ANSWER>"
        
    return response_text


def dispatch_tool(tool_name: str, tool_input: dict) -> str:
    if tool_name == "execute_python":
        code = tool_input.get("code", "")
        result = execute_python_code(code)
        logger.info(f"execute_python:\n{code}\nResult: {result[:500]}")
        return result[:10000]
    
    elif tool_name == "search_zip_corpus":
        zip_url = tool_input.get("zip_url", "")
        filename_filter = tool_input.get("filename_filter", "")
        keyword = tool_input.get("keyword", "")
        result = search_zip_corpus(zip_url, filename_filter, keyword)
        logger.info(f"search_zip_corpus(filter={filename_filter}, keyword={keyword}): {len(result)} chars")
        return result
    
    elif tool_name == "web_search":
        query = tool_input.get("query", "")
        return web_search_local(query)
    
    else:
        return f"Unknown tool: {tool_name}"


def get_llm_response(prompt: str) -> str:
    if not ANTHROPIC_AVAILABLE:
        return "<FINAL_ANSWER>Error: Anthropic SDK not available</FINAL_ANSWER>"
    
    client = anthropic.Anthropic()
    model = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
    
    tools = [
        {
            "name": "search_zip_corpus",
            "description": (
                "A perfectly general RAG tool to dynamically fetch and search a compressed corpus of files. "
                "Downloads a remote ZIP file containing text, json, or csv files into memory. "
                "You can optionally filter which files to search using filename_filter. "
                "Returns ONLY the paragraphs or lines matching your keyword to save tokens."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "zip_url": {
                        "type": "string",
                        "description": "The exact URL of the .zip file containing the corpus."
                    },
                    "filename_filter": {
                        "type": "string",
                        "description": "Optional substring to filter files (e.g., '1940' or '1953_01')."
                    },
                    "keyword": {
                        "type": "string",
                        "description": "Keyword to extract relevant paragraphs from the files. Case-insensitive."
                    }
                },
                "required": ["zip_url"]
            }
        },
        {
            "name": "execute_python",
            "description": (
                "Execute Python code for mathematical calculations. Use for regressions, "
                "standard deviations, geometric means, inflation adjustments, and any numeric computation. "
                "pandas and numpy are available. Print your final result."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute. Always print() your final answer."
                    }
                },
                "required": ["code"]
            }
        },
    ]
    
    enable_web_search = os.environ.get("ENABLE_WEB_SEARCH", "false").lower() == "true"
    if enable_web_search:
        tools.append({"type": "web_search_20250305", "name": "web_search", "max_uses": 5})
    
    messages = [{"role": "user", "content": prompt}]
    
    for step in range(15):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=messages,
                tools=tools,
                temperature=0,
            )
        except Exception as e:
            logger.error(f"Anthropic API error on step {step}: {e}")
            if "rate_limit" in str(e).lower() or "429" in str(e):
                import time
                wait_time = min(60, 2 ** step)
                logger.info(f"Retrying request to /v1/messages in {wait_time:.6f} seconds")
                time.sleep(wait_time)
                continue
            else:
                return f"Error: API failure: {e}"
        
        for block in response.content:
            if block.type == "tool_use":
                logger.info(f"Claude requested tool {block.name} with {block.input}")
                
                result = dispatch_tool(block.name, block.input)
                
                messages.append({"role": "assistant", "content": response.content})
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        }
                    ],
                })
        
        if response.stop_reason == "end_turn":
            for block in response.content:
                if block.type == "text" and "<FINAL_ANSWER>" in block.text:
                    logger.info(f"Final answer extracted:\n{block.text}")
                    return extract_final_answer(block.text)
            
            non_empty_texts = [b.text for b in response.content if b.type == "text" and b.text.strip()]
            if non_empty_texts:
                return extract_final_answer(non_empty_texts[-1])
            return "Error: Empty response"

    return "Error: max tool iterations reached"


class Executor(AgentExecutor):
    async def process_task(self, context: RequestContext, queue: EventQueue):
        logger.info(f"Processing task {context.task_id}...")
        try:
            prompt = ""
            for msg in context.messages:
                for part in msg.parts:
                    if isinstance(part, TextPart):
                        prompt += part.text + "\n"
            
            final_answer = get_llm_response(prompt)
            
            await queue.put(
                TaskStatusUpdateEvent(
                    status=TaskStatus(
                        state=TaskState.completed,
                        messages=[Message(role="assistant", parts=[TextPart(text=final_answer)])]
                    )
                )
            )
            
        except Exception as e:
            logger.error(f"Task failed: {str(e)}", exc_info=True)
            await queue.put(
                TaskStatusUpdateEvent(
                    status=TaskStatus(
                        state=TaskState.failed,
                        messages=[Message(role="assistant", parts=[TextPart(text=f"Task failed: {str(e)}")])]
                    )
                )
            )
