import asyncio
import logging
import os
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
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from google import genai
    from google.genai import types as genai_types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


SYSTEM_PROMPT = """You are an advanced financial agent designed to answer complex questions about the U.S. Treasury Bulletin. You must ensure absolute numerical accuracy.

CRITICAL INSTRUCTION ON DATA RETRIEVAL:
DO NOT rely on `web_search` to read PDFs or FRASER websites because you will hit the token limit and fail. 
Instead, use your `execute_python` tool to fetch the pre-parsed dataset CSVs directly from GitHub!
The Databricks parsed Treasury Bulletins repository is located at:
`https://api.github.com/repos/databricks/officeqa/git/trees/6aa8c32?recursive=1`
Use Python's `urllib.request` or `requests` module to fetch that tree, locate the specific `treasury_bulletins_parsed/` CSV or JSON files for the exact year/month you need, download them from `raw.githubusercontent.com` (example: `https://raw.githubusercontent.com/databricks/officeqa/6aa8c32/treasury_bulletins_parsed/your_file.csv`), and load them into a `pandas` dataframe to compute the answer entirely within python memory!

When you are completely finished, output your final answer as:
<REASONING>
[Final steps and calculations summary]
</REASONING>
<FINAL_ANSWER>
[value]
</FINAL_ANSWER>

CRITICAL INSTRUCTION ON FORMATTING: 
The <FINAL_ANSWER> tag MUST CONTAIN EXACTLY THE FINAL STRING/VALUE EXPECTED BY THE USER PROMPT. 
DO NOT WRITE CONVERSATIONAL ESSAYS LIKE "I apologize, but I'm unable to locate...". DO NOT INCLUDE ANY VERBOSE EXPLANATIONS OR HEDGING INSIDE THE <FINAL_ANSWER> TAG. IT MUST BE AS CONCISE AS HUMANLY POSSIBLE (e.g., highly specific strings like `"39482.03"` or `"36080 million"`). IF YOU PRODUCE CONVERSATIONAL TEXT INSIDE THE FINAL ANSWER TAG, YOU WILL AUTOMATICALLY FAIL THE BENCHMARK!
"""

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

def get_llm_response(prompt: str) -> str:
    provider = os.environ.get("LLM_PROVIDER", "").lower()
    use_gemini = GEMINI_AVAILABLE and (provider == "gemini" or (provider == "" and os.environ.get("GEMINI_API_KEY")))
    use_openai = OPENAI_AVAILABLE and (provider == "openai" or (provider == "" and not use_gemini and not os.environ.get("ANTHROPIC_API_KEY")))
    use_anthropic = ANTHROPIC_AVAILABLE and (provider == "anthropic" or (provider == "" and not use_openai and not use_gemini))

    messages = [{"role": "user", "content": prompt}]
    
    if use_anthropic:
        client = anthropic.Anthropic()
        model = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
        tools = [
            {
                "name": "execute_python",
                "description": "Execute python code to calculate math accurately.",
                "input_schema": {
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"]
                }
            }
        ]
        
        system_prompt = SYSTEM_PROMPT
        for step in range(10):
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                system=system_prompt,
                messages=messages,
                tools=tools,
                temperature=0,
            )
            
            messages.append({"role": "assistant", "content": response.content})
            
            if response.stop_reason == "tool_use":
                for content_block in response.content:
                    if content_block.type == "tool_use":
                        if content_block.name == "execute_python":
                            code = content_block.input["code"]
                            result = execute_python_code(code)
                            logger.info(f"Python tool executed:\n{code}\nResult: {result}")
                            messages.append({
                                "role": "user",
                                "content": [
                                    {
                                        "type": "tool_result",
                                        "tool_use_id": content_block.id,
                                        "content": result[:10000]
                                    }
                                ]
                            })

            else:
                text_parts = [b.text for b in response.content if hasattr(b, 'text')]
                return "\n".join(text_parts)

    elif use_openai:
        client = OpenAI()
        model = os.environ.get("OPENAI_MODEL", "gpt-4o")
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "execute_python",
                    "description": "Execute python code to calculate math accurately.",
                    "parameters": {
                        "type": "object",
                        "properties": {"code": {"type": "string"}},
                        "required": ["code"]
                    }
                }
            }
        ]
        
        msg_history = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        for step in range(10):
            response = client.chat.completions.create(
                model=model,
                messages=msg_history,
                tools=tools,
                temperature=0,
            )
            msg = response.choices[0].message
            msg_history.append(msg)
            
            if msg.tool_calls:
                import json
                for tc in msg.tool_calls:
                    if tc.function.name == "execute_python":
                        try:
                            args = json.loads(tc.function.arguments)
                            code = args.get("code", "")
                            result = execute_python_code(code)
                        except Exception as e:
                            result = str(e)
                            
                        logger.info(f"Python executed. Result: {result}")
                        msg_history.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result[:20000]
                        })
            else:
                return msg.content or ""

    elif use_gemini:
        client = genai.Client()
        model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        
        execute_python_tool = genai_types.Tool(
            function_declarations=[
                genai_types.FunctionDeclaration(
                    name="execute_python",
                    description="Execute python code to calculate math accurately.",
                    parameters=genai_types.Schema(
                        type=genai_types.Type.OBJECT,
                        properties={
                            "code": genai_types.Schema(type=genai_types.Type.STRING)
                        },
                        required=["code"]
                    )
                )
            ]
        )
        tools = [execute_python_tool]
        
        enable_web_search = os.environ.get("ENABLE_WEB_SEARCH", "false").lower() == "true"
        if enable_web_search:
             web_search_tool = genai_types.Tool(
                function_declarations=[
                    genai_types.FunctionDeclaration(
                        name="web_search",
                        description="AgentBeats proxy for web search."
                    )
                ]
             )
             tools.append(web_search_tool)
             
        chat = client.chats.create(
            model=model,
            config=genai_types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.0,
                tools=tools
            )
        )
        
        current_msg = prompt
        for step in range(10):
            response = chat.send_message(current_msg)
            
            if response.function_calls:
                # We need to process the tool calls and continue the loop
                tool_results = []
                for fc in response.function_calls:
                    if fc.name == "execute_python":
                        code = fc.args.get("code", "") if fc.args else ""
                        result = execute_python_code(code)
                        logger.info(f"Python tool executed:\n{code}\nResult: {result}")
                        tool_results.append(
                            genai_types.Part.from_function_response(
                                name="execute_python",
                                response={"result": result[:20000]}
                            )
                        )
                    elif fc.name == "web_search":
                        logger.warning("web_search hit local eval instead of proxy.")
                        tool_results.append(
                            genai_types.Part.from_function_response(
                                name="web_search",
                                response={"result": "Web search proxy unavailable in local testing."}
                            )
                        )
                
                # Setup the message for the next iteration
                current_msg = tool_results
            else:
                return response.text or ""

    return "<FINAL_ANSWER>Unable to determine - no LLM configured</FINAL_ANSWER>"


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
