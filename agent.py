import os
import time
import logging
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage, ToolMessage, SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI 
from openai import RateLimitError 
from dotenv import load_dotenv

from tools import (
    get_rendered_html,
    run_code,
    download_file,
    post_request,
    ocr_image_tool,
    transcribe_audio,
    encode_image_to_base64
)
from shared import url_time

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class State(TypedDict):
    messages: Annotated[list, add_messages]
    quiz_number: int
    start_time: float

SYSTEM_PROMPT = """You are an autonomous quiz-solving agent. Your mission:

1. LOAD PAGES: Use get_rendered_html to fetch quiz pages completely
2. EXTRACT INSTRUCTIONS: Parse HTML to understand exactly what's required
3. SOLVE EXACTLY: Use available tools to compute correct answers
4. SUBMIT CORRECTLY: Post answers to the exact submit endpoint with proper format
5. FOLLOW CHAIN: Continue with next_url until you receive END signal

TOOLS AVAILABLE:
- get_rendered_html: Fetch complete page content with Playwright
- run_code: Execute Python code for calculations/data processing
- download_file: Download files from URLs
- post_request: Submit answers (handles BASE64_KEY placeholders)
- ocr_image_tool: Extract text from images
- transcribe_audio: Convert audio to text
- encode_image_to_base64: Encode images and get BASE64_KEY placeholder

WORKFLOW:
1. Fetch the quiz page with get_rendered_html
2. Analyze the question, file URLs, submit endpoint
3. If files needed: download them, process with appropriate tools
4. Calculate/determine the answer
5. Submit with post_request to submit URL with email, secret, url, answer
6. If response contains next_url: repeat process
7. If response indicates END or no next_url: you're done

CRITICAL:
- Extract submit URL from page correctly
- Format answers exactly as required (number, string, JSON)
- Include email, secret, url, answer in POST payload
- Follow the URL chain until completion
- Be precise and double-check your work

Rules:
- You must ONLY output Tool Calls.
- DO NOT output conversational text.
- If you have the answer, call `post_request` IMMEDIATELY.
- **CRITICAL SUBMISSION RULE:** If "tds-llm-analysis.s-anand.net/project2" is in the url then always submit all answers to: 
  `https://tds-llm-analysis.s-anand.net/submit`
  NEVER submit to the quiz page URL itself.
  ELSE extract submit url from quiz page, it should have the key word "submit" in it
- **DATA HANDLING RULE:** NEVER fetch or read large JSON/HTML content (over 2000 chars) directly into the chat. 
  - Instead of `get_rendered_html` on a raw JSON API, use `download_file` to save it.
  - Then use `run_code` to parse and extract ONLY the specific fields you need.
  - **NEVER print the entire JSON structure.** Print only the final count or specific value needed.
- **START PAGE RULE:** If there are no clear instructions on the page e.g. ending in /project2, then just submit the email id as answer
- **IMPORTANT: If a task requires a variable (like a "cutoff"), LOOK AT THE PREVIOUS HTML OR JSON RESPONSES to find it.**
- **IMPORTANT: When using `run_code`, you MUST `print()` the final result. If you do not print it, you will not see it.**
"""

tools = [
    get_rendered_html,
    run_code,
    download_file,
    post_request,
    ocr_image_tool,
    transcribe_audio,
    encode_image_to_base64
]


llm = ChatOpenAI(
    model="gpt-4o-mini",
    base_url="https://aipipe.org/openai/v1",
    api_key=os.getenv("OPENAI_API_KEY", "dummy"),
    temperature=0
)

llm_with_tools = llm.bind_tools(tools)

# CHANGE 2: Define Gemini Fallback
llm_gemini = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  # or "gemini-1.5-pro"
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)

# Bind tools to Gemini so it can function exactly like the primary agent
llm_gemini_with_tools = llm_gemini.bind_tools(tools)


def trim_messages(messages: list, max_tokens: int = 60000) -> list:
    """Keep system prompt and recent messages within token limit"""
    if not messages:
        return messages
    
    system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
    other_msgs = [m for m in messages if not isinstance(m, SystemMessage)]
    
    # Rough estimate: 1 token ≈ 4 characters
    estimated_tokens = sum(len(str(m.content)) for m in messages) // 4
    
    if estimated_tokens <= max_tokens:
        return messages
    
    logger.warning(f"⚠️ Trimming messages: ~{estimated_tokens} tokens > {max_tokens}")
    
    # Keep last N messages that fit
    keep_count = int(len(other_msgs) * 0.6)  # Keep 60% of recent messages
    trimmed = other_msgs[-keep_count:]
    
    result = system_msgs + trimmed
    logger.info(f"✓ Trimmed to {len(result)} messages")
    return result

def agent_node(state: State) -> State:
    """Main agent reasoning node"""
    quiz_num = state["quiz_number"]
    start_time = state["start_time"]
    elapsed = time.time() - start_time
    
    logger.info("=" * 80)
    logger.info(f"🤖 AGENT NODE - Quiz #{quiz_num}")
    logger.info(f"⏱️  Time elapsed: {elapsed:.1f}s")
    logger.info("=" * 80)
    
    # Check timeout (180 seconds per quiz)
    current_url = url_time.get("current_url")
    if current_url and current_url in url_time:
        quiz_start = url_time[current_url]
        quiz_elapsed = time.time() - quiz_start
        
        if quiz_elapsed > 180:
            logger.warning(f"⏰ TIMEOUT: Quiz took {quiz_elapsed:.1f}s > 180s")
            logger.warning("🚨 Forcing wrong answer submission to move on")
            
            # Force a wrong answer submission
            messages = state["messages"]
            return {
                "messages": [AIMessage(content="TIMEOUT - submitting fallback answer")],
                "quiz_number": quiz_num,
                "start_time": start_time
            }
    
    messages = trim_messages(state["messages"])
    
    logger.info(f"📨 Sending {len(messages)} messages to LLM")
    
    try:
        # Try primary model (AIPipe)
        response = llm_with_tools.invoke(messages)
        
    except RateLimitError as e:  #
        logger.warning(f"⚠️ AIPipe Rate Limit hit! Switching to Gemini fallback. Error: {e}")
        
        # Fallback to Gemini
        # Note: We might need to filter messages if Gemini doesn't support specific system message formats,
        # but usually it handles standard LangChain messages well.
        response = llm_gemini_with_tools.invoke(messages)
        
    except Exception as e:
        # Catch-all for other token limits or API errors
        logger.error(f"❌ Primary LLM failed: {e}. Attempting fallback.")
        response = llm_gemini_with_tools.invoke(messages)

    logger.info(f"✓ LLM responded: {len(response.content) if response.content else 0} chars")
    
    if response.tool_calls:
        logger.info(f"🔧 Tool calls requested: {len(response.tool_calls)}")
        for tc in response.tool_calls:
            logger.info(f"   - {tc['name']}")
    
    return {
        "messages": [response],
        "quiz_number": quiz_num,
        "start_time": start_time
    }

def handle_malformed(state: State) -> State:
    """Handle malformed tool calls"""
    logger.warning("⚠️ MALFORMED TOOL CALL DETECTED")
    
    messages = state["messages"]
    last_message = messages[-1]
    
    tool_messages = []
    if isinstance(last_message, AIMessage) and hasattr(last_message, "tool_calls"):
        for tool_call in last_message.tool_calls:
            tool_messages.append(
                ToolMessage(
                    content="Error: Malformed tool call. Please retry with correct format.",
                    tool_call_id=tool_call["id"]
                )
            )
    
    logger.info("✓ Sending error feedback to LLM")
    return {
        "messages": tool_messages,
        "quiz_number": state["quiz_number"],
        "start_time": state["start_time"]
    }

def route(state: State) -> Literal["tools", "handle_malformed", "agent", END]:
    """Route to next node based on agent output"""
    messages = state["messages"]
    last_message = messages[-1]
    
    logger.info("🔀 ROUTING DECISION")
    
    if isinstance(last_message, AIMessage):
        # Check for malformed tool calls
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            for tool_call in last_message.tool_calls:
                if not tool_call.get("name") or not tool_call.get("args"):
                    logger.warning("   → handle_malformed (invalid tool call)")
                    return "handle_malformed"
            
            logger.info(f"   → tools ({len(last_message.tool_calls)} calls)")
            return "tools"
        
        # Check for end signals
        content = str(last_message.content).upper()
        if "END" in content or "COMPLETE" in content or "FINISHED" in content:
            logger.info("   → END (completion signal detected)")
            return END
    
    logger.info("   → agent (continue reasoning)")
    return "agent"

# Build graph
logger.info("🔨 Building LangGraph workflow")
workflow = StateGraph(State)

workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(tools))
workflow.add_node("handle_malformed", handle_malformed)

workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", route)
workflow.add_edge("tools", "agent")
workflow.add_edge("handle_malformed", "agent")

graph = workflow.compile()
logger.info("✓ Graph compiled successfully")

def run_agent(initial_url: str):
    """Main agent execution function"""
    quiz_number = 1
    current_url = initial_url
    overall_start = time.time()
    
    logger.info("\n" + "=" * 80)
    logger.info("🚀 STARTING QUIZ CHAIN")
    logger.info("=" * 80)
    logger.info(f"Initial URL: {current_url}")
    logger.info(f"Email: {os.getenv('EMAIL')}")
    logger.info("=" * 80 + "\n")
    
    while current_url:
        logger.info("\n" + "#" * 80)
        logger.info(f"# QUIZ {quiz_number}")
        logger.info(f"# URL: {current_url}")
        logger.info(f"# Time elapsed: {time.time() - overall_start:.1f}s")
        logger.info("#" * 80 + "\n")
        
        url_time["current_url"] = current_url
        url_time[current_url] = time.time()
        
        initial_state = {
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=f"Solve this quiz: {current_url}")
            ],
            "quiz_number": quiz_number,
            "start_time": time.time()
        }
        
        try:
            logger.info("▶️  Invoking agent graph")
            result = graph.invoke(
                initial_state,
                {"recursion_limit": 5000}
            )
            
            logger.info("=" * 80)
            logger.info(f"✓ QUIZ {quiz_number} PROCESSING COMPLETE")
            logger.info("=" * 80)
            
            # Check for next URL in messages
            next_url = None
            
            # Iterate backwards to find the last successful ToolMessage from post_request
            for msg in reversed(result["messages"]):
                if isinstance(msg, ToolMessage):
                    try:
                        # Parse the JSON output from the tool
                        data = json.loads(msg.content)
                        
                        # Look for 'url' OR 'next_url' key in the JSON
                        if isinstance(data, dict):
                            found_url = data.get("url") or data.get("next_url")
                            
                            # Validate it looks like a link
                            if found_url and str(found_url).startswith("http"):
                                next_url = found_url
                                logger.info(f"🔗 Next URL extracted from JSON: {next_url}")
                                break
                    except json.JSONDecodeError:
                        # Not valid JSON, skip this message
                        continue

            # Fallback: Regex check (keep this just in case)
            if not next_url:
                import re
                for msg in reversed(result["messages"]):
                    content = str(msg.content)
                    urls = re.findall(r'https?://[^\s<>"{}\\|^`\[\]]+', content)
                    # Filter out the current url to avoid loops
                    valid_urls = [u for u in urls if u != current_url]
                    if valid_urls:
                        next_url = valid_urls[0]
                        logger.info(f"🔗 Next URL extracted via Regex: {next_url}")
                        break
            # --- END OF FIX ---

            if not next_url:
                logger.info("🏁 No next URL found - chain complete")
                break
            
            current_url = next_url
            quiz_number += 1  # This will now increment correctly
            time.sleep(2)
            
        except Exception as e:
            logger.error(f"❌ Error processing quiz {quiz_number}: {str(e)}")
            logger.exception(e)
            break
    
    total_time = time.time() - overall_start
    logger.info("\n" + "=" * 80)
    logger.info("📊 QUIZ CHAIN SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total quizzes attempted: {quiz_number}")
    logger.info(f"Total time: {total_time:.1f}s")
    logger.info("=" * 80 + "\n")