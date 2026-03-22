import asyncio
import logging
import os
import re
import sys
import io as sysio
import ast
import traceback
from uuid import uuid4

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

# ── System Prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an elite Data Scientist and Python Developer answering questions about historical U.S. Treasury Bulletins.

You have access to a local file: `/app/treasury_bulletins_transformed.zip`
This ZIP file contains 697 pipe-delimited markdown text files. Files are named like `treasury_bulletin_1941_01.txt`.

DO NOT GUESS OR ESTIMATE. To answer the user's question, you must:
1. Use the `search_table_headers` tool to find exactly which files and line numbers contain the tables you need.
2. In your Python script, load the table directly into a pandas DataFrame using the injected helper function:
   `df = load_markdown_table(filename, line_number)`
   This returns a perfectly cleaned pandas DataFrame.
3. Use standard pandas operations (like `df['National defense'].sum()`) inside your `execute_python` script to calculate the mathematical answer.

CRITICAL INSTRUCTIONS FOR SPEED AND COST:
- Your token usage is strictly monitored. DO NOT print large strings! If you print >1000 characters your output gets truncated and you will fail.
- Instead of printing and reading tables manually, ALWAYS load them via `df = load_markdown_table('treasury_bulletin_1941_01.txt', 258)`.
- If you need external data (like CPI-U or inflation data), execute python to fetch it via `requests` from official sources (like FRED or BLS API).
- DO NOT print the <FINAL_ANSWER> tags from inside your Python script! 
- Once you have the correct value from your Python script output, IMMEDIATELY output it in your very next text response using:
<FINAL_ANSWER>
[value only — absolutely no words, just the number/percentage/list]
</FINAL_ANSWER>

RULES:
- Never hallucinate data. If you can't find it, adjust your table header search.
"""


# ── Corpus Indexer ─────────────────────────────────────────────────────────────
class CorpusIndexer:
    def __init__(self):
        self._toc = [] # List of dicts: {'filename': str, 'year': str, 'line_num': int, 'header': str}
        self._loaded = False
        
    def load(self):
        if self._loaded:
            return
        import zipfile
        try:
            with zipfile.ZipFile('/app/treasury_bulletins_transformed.zip', 'r') as z:
                for fname in z.namelist():
                    if not fname.endswith('.txt'): continue
                    
                    # Extract year from treasury_bulletin_1940_01.txt
                    year = "unknown"
                    parts = fname.split('_')
                    if len(parts) >= 3:
                        year = parts[2]
                        
                    content = z.read(fname).decode('utf-8', errors='replace')
                    lines = content.split('\n')
                    
                    for i, line in enumerate(lines):
                        # Very simple table detection
                        if "|" in line and "---" in line and i > 0:
                            header = lines[i-1].strip()
                            title = lines[i-2].strip().replace("|", "") if i > 1 else ""
                            full_header = f"{title} {header}".strip()
                            if full_header:
                                self._toc.append({
                                    'filename': fname,
                                    'year': year,
                                    'line_num': i-1,
                                    'header': full_header
                                })
            self._loaded = True
            logger.info(f"CorpusIndexer indexed {len(self._toc)} tables.")
        except Exception as e:
            logger.error(f"Failed to load ZIP: {e}")

    def search(self, keyword: str, year: str = None) -> str:
        self.load()
        results = []
        kw = keyword.lower()
        for t in self._toc:
            if kw in t['header'].lower():
                if year is None or year in t['year'] or str(int(year)+1) in t['year']:
                    results.append(f"File: {t['filename']} | Line: {t['line_num']} | Header: {t['header']}")
                    
        if not results:
            return f"No tables found matching '{keyword}' for year '{year}'."
            
        # Deduplicate identical consecutive headers in same file
        dedup = []
        last = ""
        for r in results:
            if r != last:
                dedup.append(r)
                last = r
                
        out = "\n".join(dedup[:30])
        if len(dedup) > 30:
            out += f"\n...and {len(dedup)-30} more. Refine your search."
        return out

_indexer = CorpusIndexer()

# ── Tools ──────────────────────────────────────────────────────────────────────
import zipfile
import io
import pandas as pd

def __load_markdown_table_to_df(filename: str, line_number: int) -> pd.DataFrame:
    with zipfile.ZipFile('/app/treasury_bulletins_transformed.zip', 'r') as z:
        content = z.read(filename).decode('utf-8', errors='replace')
        
    lines = content.split('\n')
    if line_number < 0 or line_number >= len(lines):
        raise ValueError(f"Line number {line_number} out of bounds.")
        
    # Find exact header row around the provided line number
    header_idx = line_number
    for i in range(max(0, line_number-2), min(line_number+5, len(lines)-1)):
        if "|" in lines[i] and "---" in lines[i+1]:
            header_idx = i
            break
            
    table_lines = [lines[header_idx]]
    for i in range(header_idx + 2, len(lines)):
        row = lines[i]
        if "|" not in row or row.strip() == "":
            break
        table_lines.append(row)
        
    csv_str = "\n".join(table_lines)
    df = pd.read_csv(io.StringIO(csv_str), sep="|", skipinitialspace=True)
    
    # Drop first and last empty columns from markdown leading/trailing pipes
    if df.columns[0].strip() == "" or 'Unnamed: 0' in df.columns[0]:
        df = df.iloc[:, 1:]
    if len(df.columns) > 0 and (df.columns[-1].strip() == "" or 'Unnamed' in df.columns[-1]):
        df = df.iloc[:, :-1]
        
    df.columns = df.columns.str.strip()
    for col in df.columns:
        if df[col].dtype == 'object':
             try: df[col] = df[col].str.strip()
             except: pass
    
    # Remove any internal row that acts as a secondary header/separator (often filled with whitespace or NaNs)
    df = df.dropna(how='all')
    return df

def execute_python_code(code: str, state: dict) -> str:
    import traceback
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout
    
    try:
        if 'load_markdown_table' not in state:
            state['load_markdown_table'] = __load_markdown_table_to_df
            import pandas as pd
            state['pd'] = pd

        exec(code, state, state)
        output = new_stdout.getvalue()
        if not output.strip():
            return "Code executed successfully (no printed output)."
        return output
    except Exception:
        return "Exception occurred:\n" + traceback.format_exc()
    finally:
        sys.stdout = old_stdout


def extract_final_answer(text: str) -> str:
    match = re.search(r"<FINAL_ANSWER>(.*?)</FINAL_ANSWER>", text, re.DOTALL)
    if match:
        answer = match.group(1).strip()
        for prefix in ["The answer is ", "The answer is: ", "Answer: ",
                        "The value is ", "The result is ", "Result: "]:
            if answer.lower().startswith(prefix.lower()):
                answer = answer[len(prefix):].strip()
        return f"<FINAL_ANSWER>\n{answer}\n</FINAL_ANSWER>"
    if len(text.split()) < 20:
        return f"<FINAL_ANSWER>\n{text.strip()}\n</FINAL_ANSWER>"
    return text


# ── Main LLM Call ──────────────────────────────────────────────────────────────
def get_llm_response(question: str) -> str:
    if not ANTHROPIC_AVAILABLE:
        return "<FINAL_ANSWER>Error: Anthropic SDK not available</FINAL_ANSWER>"

    client = anthropic.Anthropic()
    model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    
    global_python_state = {}

    tools = [
        {
            "name": "search_table_headers",
            "description": "Quickly search all 697 Treasury Bulletins for tables containing a specific keyword in their title or header.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": "e.g., 'defense', 'veteran', 'expenditure'"},
                    "year": {"type": "string", "description": "Optional. Limit search to files from this year and the following year (e.g., '1940' searches 1940 and 1941)."}
                },
                "required": ["keyword"]
            }
        },
        {
            "name": "execute_python",
            "description": "Run Python code. pandas/numpy/requests/zipfile available. Scripts run in /app. Always print() results.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to run."}
                },
                "required": ["code"]
            }
        },
    ]

    messages = [{"role": "user", "content": f"QUESTION: {question}"}]

    for step in range(45):
        # Rolling Context Compression: Truncate large tool_results strictly older than the last 4 messages
        if len(messages) > 6:
            for i in range(1, len(messages) - 4):
                if messages[i].get("role") == "user" and isinstance(messages[i].get("content"), list):
                    for block in messages[i]["content"]:
                        if block.get("type") == "tool_result" and isinstance(block.get("content"), str):
                            text = block["content"]
                            if len(text) > 150:
                                block["content"] = "[Output truncated to save tokens] " + text[:50] + "..." + text[-50:]

        try:
            response = client.messages.create(
                model=model,
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                messages=messages,
                tools=tools,
                temperature=0,
            )
        except Exception as e:
            logger.error(f"API error step {step}: {e}")
            if "rate_limit" in str(e).lower() or "429" in str(e):
                import time
                time.sleep(min(30, 2 ** step))
                continue
            return f"Error: {e}"

        # Immediately check if Claude outputted the FINAL_ANSWER in any text block
        for block in response.content:
            if block.type == "text" and "<FINAL_ANSWER>" in block.text:
                return extract_final_answer(block.text)

        messages.append({"role": "assistant", "content": response.content})

        tool_called = False
        tool_results = []
        for block in response.content:
            if block.type == "text" and block.text.strip():
                logger.info(f"CLAUDE THINKING: {block.text.strip()[:500]}")
                
            if block.type == "tool_use":
                logger.info(f"CLAUDE TOOL: {block.name} with {block.input}")
                tool_called = True
                result = ""
                
                if block.name == "search_table_headers":
                    kw = block.input.get("keyword", "")
                    yr = block.input.get("year")
                    result = _indexer.search(kw, yr)
                elif block.name == "execute_python":
                    code = block.input.get("code", "")
                    result = execute_python_code(code, global_python_state)
                    if len(result) > 1000:
                        result = result[:1000] + "\n[...printed output truncated due to 1000 char token limit...]"
                else:
                    result = f"Unknown tool: {block.name}"

                tool_results.append({
                    "type": "tool_result", 
                    "tool_use_id": block.id, 
                    "content": result
                })
                
        if tool_called and tool_results:
            messages.append({
                "role": "user",
                "content": tool_results
            })

        if not tool_called and response.stop_reason == "end_turn":
            for block in response.content:
                if block.type == "text" and "<FINAL_ANSWER>" in block.text:
                    return extract_final_answer(block.text)
            texts = [b.text for b in response.content if b.type == "text" and b.text.strip()]
            if texts:
                return extract_final_answer(texts[-1])
            return "Error: Empty response"

    return "Error: max iterations reached"


# ── A2A Executor ───────────────────────────────────────────────────────────────
class Executor(AgentExecutor):
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
                        parts=[Part(root=TextPart(kind="text", text="Processing..."))],
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
            response = f"<FINAL_ANSWER>Error: {e}</FINAL_ANSWER>"

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
