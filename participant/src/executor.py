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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GITHUB_TREE_URL = (
    "https://api.github.com/repos/databricks/officeqa/git/trees/6aa8c32?recursive=1"
)
RAW_BASE = (
    "https://raw.githubusercontent.com/databricks/officeqa/6aa8c32/"
)

# ---------------------------------------------------------------------------
# SYSTEM PROMPT — instructs the LLM to use the custom tools
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an advanced financial agent designed to answer complex questions about the U.S. Treasury Bulletin. You must ensure absolute numerical accuracy.

You have access to three tools:

1. **query_treasury_corpus** — Use this FIRST to retrieve relevant Treasury Bulletin data.
   Supply `year` (string like "1940"), `month` (string like "January" or "" for annual), and `keyword` (string to filter rows, e.g. "national defense").
   The tool returns matching CSV rows as text. If no rows match, try broadening your keyword or checking adjacent months/years.

2. **execute_python** — Use this for ALL math: regressions, standard deviations, geometric means, inflation adjustments, etc.
   Write complete Python scripts. Use pandas and numpy as needed — they are pre-installed.
   You can pass the data you received from query_treasury_corpus into your Python code.

3. **web_search** — Use this as a LAST RESORT for data not found in the Treasury corpus (e.g., CPI-U values from FRED, exchange rates from Macrotrends, etc.).

WORKFLOW:
1. Analyze the question to identify the year(s), month(s), and category/keyword.
2. Call query_treasury_corpus to retrieve the relevant data rows.
3. If the data needs math, call execute_python with the retrieved values.
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

# ---------------------------------------------------------------------------
# Tool 1: Treasury Corpus RAG — dynamic GitHub fetching + pandas filtering
# ---------------------------------------------------------------------------
_tree_cache = None

def _get_corpus_tree():
    """Fetch and cache the GitHub tree listing of all parsed Treasury files."""
    global _tree_cache
    if _tree_cache is not None:
        return _tree_cache
    
    import requests
    try:
        resp = requests.get(GITHUB_TREE_URL, timeout=30)
        resp.raise_for_status()
        tree_data = resp.json()
        # Extract just the paths under treasury_bulletins_parsed/
        paths = [
            item["path"]
            for item in tree_data.get("tree", [])
            if item["path"].startswith("treasury_bulletins_parsed/")
            and item["type"] == "blob"
        ]
        _tree_cache = paths
        logger.info(f"Loaded corpus tree with {len(paths)} files.")
        return paths
    except Exception as e:
        logger.error(f"Failed to fetch corpus tree: {e}")
        return []


def _find_matching_files(year: str, month: str, paths: list) -> list:
    """Find files in the corpus that match the given year and optionally month."""
    year = year.strip().lower()
    month = month.strip().lower() if month else ""
    
    matches = []
    for p in paths:
        p_lower = p.lower()
        if year in p_lower:
            if month:
                if month in p_lower:
                    matches.append(p)
            else:
                matches.append(p)
    
    # If too many matches, prefer CSV files
    if len(matches) > 20:
        csv_matches = [m for m in matches if m.endswith('.csv')]
        if csv_matches:
            matches = csv_matches[:20]
        else:
            matches = matches[:20]
    
    return matches


def query_treasury_corpus(year: str, month: str = "", keyword: str = "") -> str:
    """
    Dynamically fetch parsed Treasury Bulletin data from the Databricks GitHub repo.
    Filters rows by keyword and returns concise text to the LLM.
    """
    import requests
    import io
    
    paths = _get_corpus_tree()
    if not paths:
        return "Error: Could not fetch corpus file listing from GitHub."
    
    matching_files = _find_matching_files(year, month, paths)
    if not matching_files:
        # Try without month
        matching_files = _find_matching_files(year, "", paths)
    
    if not matching_files:
        return f"No files found for year={year}, month={month}. Available years in corpus: try browsing with execute_python."
    
    results = []
    keyword_lower = keyword.strip().lower() if keyword else ""
    
    for fpath in matching_files[:5]:  # Limit to 5 files to control token usage
        url = RAW_BASE + fpath
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            content = resp.text
            
            fname = fpath.split("/")[-1]
            
            if fpath.endswith(".csv"):
                try:
                    import pandas as pd
                    df = pd.read_csv(io.StringIO(content))
                    
                    if keyword_lower and len(df) > 0:
                        # Filter rows where any column contains the keyword
                        mask = df.apply(
                            lambda row: any(
                                keyword_lower in str(v).lower() for v in row
                            ),
                            axis=1,
                        )
                        filtered = df[mask]
                        if len(filtered) > 0:
                            # Return filtered rows as compact string
                            result_str = f"\n--- {fname} (filtered by '{keyword}') ---\n"
                            result_str += f"Columns: {list(df.columns)}\n"
                            result_str += filtered.head(50).to_string(index=False)
                            results.append(result_str)
                        else:
                            # Return column names and first few rows as context
                            result_str = f"\n--- {fname} (no rows match '{keyword}', showing headers + sample) ---\n"
                            result_str += f"Columns: {list(df.columns)}\n"
                            result_str += df.head(5).to_string(index=False)
                            results.append(result_str)
                    else:
                        # No keyword filter — return structure + sample
                        result_str = f"\n--- {fname} ---\n"
                        result_str += f"Shape: {df.shape}, Columns: {list(df.columns)}\n"
                        result_str += df.head(20).to_string(index=False)
                        results.append(result_str)
                except Exception as e:
                    results.append(f"\n--- {fname} (CSV parse error: {e}) ---")
            else:
                # For non-CSV (text/json), return a truncated snippet
                snippet = content[:3000]
                results.append(f"\n--- {fname} ---\n{snippet}")
        except Exception as e:
            results.append(f"\n--- {fpath} (fetch error: {e}) ---")
    
    if not results:
        return f"Found {len(matching_files)} files but could not extract data."
    
    output = "\n".join(results)
    # Hard cap to prevent token explosion
    if len(output) > 8000:
        output = output[:8000] + "\n... [truncated]"
    return output


# ---------------------------------------------------------------------------
# Tool 2: Execute Python — for all math/computation
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Tool 3: Web search (proxy) — last resort for external data
# ---------------------------------------------------------------------------
def web_search_local(query: str) -> str:
    """Fallback web search — in local testing this returns a stub."""
    logger.warning(f"web_search called locally with query: {query}")
    return "Web search proxy unavailable in local testing. Use query_treasury_corpus or execute_python instead."


# ---------------------------------------------------------------------------
# Strict Regex Output Enforcer
# ---------------------------------------------------------------------------
def extract_final_answer(response_text: str) -> str:
    """
    Extract ONLY the content inside <FINAL_ANSWER>...</FINAL_ANSWER> tags.
    Strips all conversational padding. Falls back to raw response if no tags found.
    """
    match = re.search(r"<FINAL_ANSWER>(.*?)</FINAL_ANSWER>", response_text, re.DOTALL)
    if match:
        answer = match.group(1).strip()
        # Additional cleanup: remove common conversational prefixes
        prefixes_to_strip = [
            "The answer is ", "The answer is: ", "Answer: ",
            "The value is ", "The result is ", "Result: ",
        ]
        for prefix in prefixes_to_strip:
            if answer.lower().startswith(prefix.lower()):
                answer = answer[len(prefix):].strip()
        return answer
    
    # No FINAL_ANSWER tags found — return the response as-is (the framework will handle it)
    return response_text


# ---------------------------------------------------------------------------
# Dispatch tool calls
# ---------------------------------------------------------------------------
def dispatch_tool(tool_name: str, tool_input: dict) -> str:
    """Route a tool call to the appropriate handler."""
    if tool_name == "execute_python":
        code = tool_input.get("code", "")
        result = execute_python_code(code)
        logger.info(f"execute_python:\n{code}\nResult: {result[:500]}")
        return result[:10000]
    
    elif tool_name == "query_treasury_corpus":
        year = tool_input.get("year", "")
        month = tool_input.get("month", "")
        keyword = tool_input.get("keyword", "")
        result = query_treasury_corpus(year, month, keyword)
        logger.info(f"query_treasury_corpus(year={year}, month={month}, keyword={keyword}): {len(result)} chars")
        return result
    
    elif tool_name == "web_search":
        query = tool_input.get("query", "")
        return web_search_local(query)
    
    else:
        return f"Unknown tool: {tool_name}"


# ---------------------------------------------------------------------------
# Main LLM response function
# ---------------------------------------------------------------------------
def get_llm_response(prompt: str) -> str:
    if not ANTHROPIC_AVAILABLE:
        return "<FINAL_ANSWER>Error: Anthropic SDK not available</FINAL_ANSWER>"
    
    client = anthropic.Anthropic()
    model = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
    
    # Register all three tools with Claude
    tools = [
        {
            "name": "query_treasury_corpus",
            "description": (
                "Retrieve parsed U.S. Treasury Bulletin data for a specific year and optional month. "
                "Returns matching CSV rows filtered by keyword. Use this FIRST to find financial data "
                "before doing any calculations. The corpus contains 697 parsed bulletin documents."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "year": {
                        "type": "string",
                        "description": "The year to search for, e.g. '1940' or '1981'"
                    },
                    "month": {
                        "type": "string",
                        "description": "Optional month name, e.g. 'January', 'March'. Leave empty for annual data."
                    },
                    "keyword": {
                        "type": "string",
                        "description": "Keyword to filter rows, e.g. 'national defense', 'customs', 'interest'. Case-insensitive."
                    }
                },
                "required": ["year"]
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
    
    # Optionally add web_search (proxied by AgentBeats in competition)
    enable_web_search = os.environ.get("ENABLE_WEB_SEARCH", "false").lower() == "true"
    if enable_web_search:
        tools.append({"type": "web_search_20250305", "name": "web_search", "max_uses": 5})
    
    messages = [{"role": "user", "content": prompt}]
    
    # Agentic loop — up to 15 iterations of tool use
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
                time.sleep(5)  # Brief backoff for rate limits
                continue
            return f"<FINAL_ANSWER>Error: {e}</FINAL_ANSWER>"
        
        messages.append({"role": "assistant", "content": response.content})
        
        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = dispatch_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
            messages.append({"role": "user", "content": tool_results})
        else:
            # End of generation — extract text and enforce formatting
            text_parts = [b.text for b in response.content if hasattr(b, "text")]
            raw_response = "\n".join(text_parts)
            
            # Apply strict regex enforcer
            final_answer = extract_final_answer(raw_response)
            logger.info(f"Final answer extracted: {final_answer[:200]}")
            return final_answer
    
    return "<FINAL_ANSWER>Error: max tool iterations reached</FINAL_ANSWER>"


# ---------------------------------------------------------------------------
# A2A Executor class (unchanged interface)
# ---------------------------------------------------------------------------
class Executor(AgentExecutor):
    def __init__(self):
        self._contexts: dict[str, list[dict]] = {}

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        message = context.message
        if not message or not message.parts:
            logger.warning("Received empty message")
            return

        task = context.current_task
        if task and task.status.state in TERMINAL_STATES:
            logger.info(f"Task {task.id} already in terminal state")
            return

        task_id = context.task_id or "unknown"
        context_id = context.context_id or "unknown"

        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                taskId=task_id,
                contextId=context_id,
                status=TaskStatus(
                    state=TaskState.working,
                    message=Message(
                        messageId=uuid4().hex,
                        role="agent",
                        parts=[Part(root=TextPart(kind="text", text="Processing question..."))],
                    ),
                ),
                final=False,
            )
        )

        question_text = ""
        for part in message.parts:
            root = part.root if hasattr(part, 'root') else part
            if isinstance(root, TextPart):
                question_text = root.text
                break

        try:
            response = await asyncio.to_thread(get_llm_response, question_text)
        except Exception as e:
            logger.exception(f"LLM call failed: {e}")
            response = f"Error: {e}"

        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                taskId=task_id,
                contextId=context_id,
                status=TaskStatus(
                    state=TaskState.completed,
                    message=Message(
                        messageId=uuid4().hex,
                        role="agent",
                        parts=[Part(root=TextPart(kind="text", text=response))],
                    ),
                ),
                final=True,
            )
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise UnsupportedOperationError(message="Cancellation not supported")
