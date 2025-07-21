"""Node definitions for the LangGraph workflow."""

import json
from typing import Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage

from ..config.settings import (
    SYSTEM_PROMPT,
    GEMINI_API_KEY,
    GEMINI_MODEL_CONFIG,
    logging,
    google_exceptions
)
from .state import InteractivePlanState
from langchain_google_genai import ChatGoogleGenerativeAI

# Initialize LLM
llm = None
if not GEMINI_API_KEY:
    logging.error("Gemini API Key is missing or invalid. Cannot initialize LLM.")
else:
    try:
        llm = ChatGoogleGenerativeAI(
            google_api_key=GEMINI_API_KEY,
            **GEMINI_MODEL_CONFIG
        )
        logging.info("ChatGoogleGenerativeAI model initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize ChatGoogleGenerativeAI: {e}", exc_info=True)

def planner_agent_node(state: InteractivePlanState) -> Dict[str, Any]:
    """
    Conversational planner node. Invokes LLM with history and tools.
    Decides whether to ask questions, call tools, generate/revise plan.
    """
    logging.info("--- Running Node: planner_agent_node ---")
    if llm is None:
        return {"error_message": "LLM client failed to initialize."}

    messages = state['messages']
    logging.info(f"Planner received {len(messages)} messages. Last: {type(messages[-1])} - {str(messages[-1].content)[:100]}...")

    # Add the system prompt if it's not already the first message
    current_messages = list(messages)
    if not current_messages or not isinstance(current_messages[0], SystemMessage):
        current_messages.insert(0, SystemMessage(content=SYSTEM_PROMPT))

    from ..utils.tools import tools  # Import here to avoid circular dependency
    llm_with_tools = llm.bind_tools(tools)

    try:
        # Invoke LLM with the full conversation history + system prompt
        ai_response = llm_with_tools.invoke(current_messages)
        logging.info(f"LLM Response: Type: {type(ai_response)}")
        if isinstance(ai_response.content, str):
            logging.info(f"LLM Response content snippet: {ai_response.content[:200]}...")
        if ai_response.tool_calls:
            logging.info(f"LLM Response tool calls: {ai_response.tool_calls}")

        return {"messages": [ai_response], "error_message": None}

    except google_exceptions.ResourceExhausted as e:
         logging.error(f"LLM API quota exceeded: {e}")
         return {
             "messages": [AIMessage(content="Sorry, I encountered an API limit. Please try again later.")],
             "error_message": "Gemini API quota likely exceeded."
         }
    except Exception as e:
        logging.error(f"LLM invocation failed: {e}", exc_info=True)
        return {
            "messages": [AIMessage(content=f"Sorry, an internal error occurred: {e}")],
            "error_message": f"LLM Error: {e}"
        }

def tool_executor_node(state: InteractivePlanState) -> Dict[str, Any]:
    """Executes tools called by the Planner Agent."""
    logging.info("--- Running Node: tool_executor_node ---")
    messages = list(state['messages'])
    last_message = messages[-1]

    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        logging.warning("Tool executor called, but last message has no tool calls.")
        return {}

    tool_calls = last_message.tool_calls
    logging.info(f"Executing {len(tool_calls)} tool calls: {[tc.get('name') for tc in tool_calls]}")

    from ..utils.tools import tools  # Import here to avoid circular dependency
    tool_messages = []
    available_tools_map = {t.name: t for t in tools}

    for tool_call in tool_calls:
        tool_name = tool_call.get('name')
        tool_args = tool_call.get('args', {})
        tool_call_id = tool_call.get('id')

        if not tool_call_id:
             logging.error(f"Tool call missing 'id': {tool_call}")
             tool_messages.append(ToolMessage(
                 content=json.dumps({"error": "Tool call missing 'id'."}),
                 tool_call_id=f"error_missing_id_{tool_name}"
             ))
             continue

        if tool_name in available_tools_map:
            selected_tool = available_tools_map[tool_name]
            try:
                logging.info(f"Invoking tool: {tool_name} with args: {tool_args}")
                output = selected_tool.invoke(tool_args)
                try:
                    output_content = json.dumps(output)
                    logging.info(f"Tool '{tool_name}' executed successfully. Output snippet: {output_content[:200]}...")
                except TypeError as e:
                     logging.error(f"Tool '{tool_name}' output is not JSON serializable: {e}. Output: {output}")
                     output_content = json.dumps({
                         "error": f"Tool output serialization failed: {e}",
                         "output_type": str(type(output))
                     })
                tool_messages.append(ToolMessage(content=output_content, tool_call_id=tool_call_id))
            except Exception as e:
                logging.error(f"Error executing tool '{tool_name}': {e}", exc_info=True)
                tool_messages.append(ToolMessage(
                    content=json.dumps({"error": f"Execution failed: {e}"}),
                    tool_call_id=tool_call_id
                ))
        else:
            logging.warning(f"LLM called unknown tool: '{tool_name}'")
            tool_messages.append(ToolMessage(
                content=json.dumps({"error": f"Unknown tool '{tool_name}' called."}),
                tool_call_id=tool_call_id
            ))

    return {"messages": tool_messages, "error_message": None}

def parse_and_save_plan_node(state: InteractivePlanState) -> Dict[str, Any]:
    """
    Parses potential JSON plan from the last AI message and updates the state.
    """
    logging.info("--- Running Node: parse_and_save_plan_node ---")
    messages = state.get('messages', [])
    last_ai_message = messages[-1] if messages and isinstance(messages[-1], AIMessage) else None

    parsed_plan = None
    parsing_error = None
    content_to_parse = None

    if not last_ai_message or last_ai_message.tool_calls:
        logging.info("PARSE_DEBUG: Last message is not AI or has tool calls. No plan to parse.")
        return {}

    ai_content = last_ai_message.content
    logging.info(f"PARSE_DEBUG: AI Message Content Type: {type(ai_content)}")

    # Find the content containing the JSON
    if isinstance(ai_content, str):
        if '"itinerary"' in ai_content and (ai_content.strip().startswith('{') or ai_content.strip().startswith('```json')):
            content_to_parse = ai_content
    elif isinstance(ai_content, list):
        for item in ai_content:
            if isinstance(item, str) and '"itinerary"' in item and (item.strip().startswith('{') or item.strip().startswith('```json')):
                content_to_parse = item
                break

    if not content_to_parse:
        if state.get("error_message"):
             return {"error_message": None}
        return {}

    # Try parsing the identified content
    try:
        json_str = content_to_parse
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0].strip()
        elif not json_str.strip().startswith('{'):
             start_index = json_str.find('{')
             end_index = json_str.rfind('}')
             if start_index != -1 and end_index != -1 and start_index < end_index:
                 json_str = json_str[start_index:end_index+1]
             else:
                 raise json.JSONDecodeError("No clear JSON object boundaries found", json_str, 0)

        parsed_plan_candidate = json.loads(json_str)

        # Basic validation
        if isinstance(parsed_plan_candidate, dict) and "itinerary" in parsed_plan_candidate and isinstance(parsed_plan_candidate["itinerary"], (dict, list)):
             parsed_plan = parsed_plan_candidate
        else:
             parsed_plan = None
             parsing_error = "Parsed JSON has incorrect structure."

    except json.JSONDecodeError as e:
        parsed_plan = None
        parsing_error = f"Failed to parse final plan JSON: {e}"
    except Exception as e:
        parsed_plan = None
        parsing_error = f"Unexpected error parsing plan: {e}"

    # Return update for state
    update = {"current_plan": parsed_plan}
    if parsing_error:
         update["error_message"] = parsing_error
    elif parsed_plan is not None and state.get("error_message"):
         update["error_message"] = None

    return update
