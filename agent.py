from langgraph.graph import StateGraph, END, START
from shared_store import url_time
import time
from langchain_core.rate_limiters import InMemoryRateLimiter
from langgraph.prebuilt import ToolNode
from tools import (
    get_rendered_html, download_file, post_request,
    run_code, add_dependencies, ocr_image_tool, transcribe_audio, encode_image_to_base64
)
from typing import TypedDict, Annotated, List
from langchain_core.messages import trim_messages, HumanMessage
from langchain.chat_models import init_chat_model
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
import os
from dotenv import load_dotenv
load_dotenv()

EMAIL = os.getenv("EMAIL")
SECRET = os.getenv("SECRET")

RECURSION_LIMIT = 5000
MAX_TOKENS = 60000


# -------------------------------------------------
# STATE
# -------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[List, add_messages]


TOOLS = [
    run_code, get_rendered_html, download_file,
    post_request, add_dependencies, ocr_image_tool, transcribe_audio, encode_image_to_base64
]


# -------------------------------------------------
# LLM INIT
# -------------------------------------------------
rate_limiter = InMemoryRateLimiter(
    requests_per_second=15 / 60,
    check_every_n_seconds=0.1,
    max_bucket_size=1
)

llm = init_chat_model(
    model_provider="google_genai",
    model="gemini-2.5-flash",
    rate_limiter=rate_limiter
).bind_tools(TOOLS)


# -------------------------------------------------
# SYSTEM PROMPT
# -------------------------------------------------
SYSTEM_PROMPT = f"""
You are an autonomous quiz-solving agent.

Your job is to:
1. Load each quiz page from the given URL.
2. Extract instructions, parameters, and submit endpoint.
3. Solve tasks exactly.
4. Submit answers ONLY to the correct endpoint.
5. Follow new URLs until none remain, then output END.

Rules:
- For base64 generation of an image NEVER use your own code, always use the "encode_image_to_base64" tool that's provided
- Never hallucinate URLs or fields.
- Never shorten endpoints.
- Always inspect server response.
- Never stop early.
- Use tools for HTML, downloading, rendering, OCR, or running code.
- Include:
    email = {EMAIL}
    secret = {SECRET}
"""


# -------------------------------------------------
# NEW NODE: HANDLE MALFORMED JSON
# -------------------------------------------------
def handle_malformed_node(state: AgentState):
    """
    If the LLM generates invalid JSON, this node sends a correction message
    so the LLM can try again.
    """
    print("--- DETECTED MALFORMED JSON. ASKING AGENT TO RETRY ---")
    return {
        "messages": [
            {
                "role": "user", 
                "content": "SYSTEM ERROR: Your last tool call was Malformed (Invalid JSON). Please rewrite the code and try again. Ensure you escape newlines and quotes correctly inside the JSON."
            }
        ]
    }


# -------------------------------------------------
# AGENT NODE
# -------------------------------------------------
# -------------------------------------------------
# AGENT NODE (WITH SLIDING WINDOW MEMORY)
# -------------------------------------------------
def agent_node(state: AgentState):
    # --- TIME TRACKING & LOGGING ---
    cur_time = time.time()
    cur_url = os.getenv("url", "Unknown URL")
    prev_time = url_time.get(cur_url)
    offset = os.getenv("offset", "0")

    if prev_time is not None:
        diff = cur_time - float(prev_time)
        print(f"\n⏱️  TIMER: {diff:.1f}s / 180s (Task: {cur_url})")
        
        if diff >= 180 or (offset != "0" and (cur_time - float(offset)) > 90):
            print(f"!!! TIMEOUT EXCEEDED ({diff:.1f}s) - FORCING WRONG ANSWER !!!")
            fail_msg = HumanMessage(content="You have exceeded the time limit (180s). Immediately call `post_request` with a WRONG answer to skip.")
            # Urgent invoke to handle the timeout immediately
            return {"messages": [llm.invoke(state["messages"] + [fail_msg])]}
    # -------------------------------


    # --- MEMORY OPTIMIZATION (THE FIX) ---
    all_messages = state["messages"]
    
    # 1. Identify System Prompt (Always index 0)
    system_prompt = all_messages[0]
    
    # 2. Get the conversation history (excluding system prompt)
    history = all_messages[1:]
    
    # 3. SLIDING WINDOW: Keep only the last 10 messages
    # This is plenty for solving ONE quiz, but forgets the previous completed ones.
    if len(history) > 15:
        print(f"✂️  Trimming history: Dropping {len(history) - 15} old messages...")
        history = history[-15:]
        
        # 4. SAFETY: Ensure we don't start with a 'ToolMessage'
        # (A ToolMessage must always follow an AIMessage. If we cut the parent, we must cut the child.)
        while len(history) > 0 and isinstance(history[0], (ToolMessage, FunctionMessage)):
            history.pop(0)

    # Reconstruct: System Prompt + Recent History
    trimmed_messages = [system_prompt] + history
    state["messages"] = trimmed_messages
    # -------------------------------------


    print(f"--- INVOKING AGENT (Context: {len(trimmed_messages)} items) ---")
    
    # Generate response
    response = llm.invoke(trimmed_messages)
    
    return {"messages": [response]}


# -------------------------------------------------
# ROUTE LOGIC (UPDATED FOR MALFORMED CALLS)
# -------------------------------------------------
def route(state):
    last = state["messages"][-1]
    
    # 1. CHECK FOR MALFORMED FUNCTION CALLS
    if "finish_reason" in last.response_metadata:
        if last.response_metadata["finish_reason"] == "MALFORMED_FUNCTION_CALL":
            return "handle_malformed"

    # 2. CHECK FOR VALID TOOLS
    tool_calls = getattr(last, "tool_calls", None)
    if tool_calls:
        print("Route → tools")
        return "tools"

    # 3. CHECK FOR END
    content = getattr(last, "content", None)
    if isinstance(content, str) and content.strip() == "END":
        return END

    if isinstance(content, list) and len(content) and isinstance(content[0], dict):
        if content[0].get("text", "").strip() == "END":
            return END

    print("Route → agent")
    return "agent"


# -------------------------------------------------
# GRAPH
# -------------------------------------------------
graph = StateGraph(AgentState)

# Add Nodes
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(TOOLS))
graph.add_node("handle_malformed", handle_malformed_node) # Add the repair node

# Add Edges
graph.add_edge(START, "agent")
graph.add_edge("tools", "agent")
graph.add_edge("handle_malformed", "agent") # Retry loop

# Conditional Edges
graph.add_conditional_edges(
    "agent", 
    route,
    {
        "tools": "tools",
        "agent": "agent",
        "handle_malformed": "handle_malformed", # Map the new route
        END: END
    }
)

app = graph.compile()


# -------------------------------------------------
# RUNNER
# -------------------------------------------------
def run_agent(url: str):
    # system message is seeded ONCE here
    initial_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": url}
    ]

    app.invoke(
        {"messages": initial_messages},
        config={"recursion_limit": RECURSION_LIMIT}
    )

    print("Tasks completed successfully!")
