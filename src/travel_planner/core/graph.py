"""LangGraph workflow definition for the travel planning system."""

from langgraph.graph import StateGraph, END
from typing import Dict, Any
from langchain_core.messages import AIMessage

from ..config.settings import logging
from .state import InteractivePlanState
from .nodes import planner_agent_node, tool_executor_node, parse_and_save_plan_node

def route_after_planner(state: InteractivePlanState) -> str:
    """Routes from the planner based on the AI's response."""
    messages = state['messages']
    last_message = messages[-1] if messages else None

    if isinstance(last_message, AIMessage):
        if last_message.tool_calls:
            logging.info("Conditional Edge: Routing to tool executor.")
            return "call_tools"
        else:
            logging.info("Conditional Edge: Routing to parse/save plan.")
            return "parse_plan"
    else:
        logging.warning(f"Conditional Edge: Unexpected message type after planner ({type(last_message)}). Routing to END.")
        return "error_end"

def create_graph() -> StateGraph:
    """Creates and returns the LangGraph workflow."""
    workflow = StateGraph(InteractivePlanState)

    # Add nodes
    workflow.add_node("planner_agent", planner_agent_node)
    workflow.add_node("tool_executor", tool_executor_node)
    workflow.add_node("parse_and_save_plan", parse_and_save_plan_node)

    # Define edges
    workflow.set_entry_point("planner_agent")
    workflow.add_edge("tool_executor", "planner_agent")

    # Conditional edge from planner
    workflow.add_conditional_edges(
        "planner_agent",
        route_after_planner,
        {
            "call_tools": "tool_executor",
            "parse_plan": "parse_and_save_plan",
            "error_end": END
        }
    )

    # Edge from parsing node to END
    workflow.add_edge("parse_and_save_plan", END)

    return workflow

def compile_graph() -> Any:
    """Compiles and returns the LangGraph application."""
    workflow = create_graph()
    try:
        app = workflow.compile()
        logging.info("Interactive LangGraph compiled successfully.")
        return app
    except Exception as compile_error:
        logging.error(f"Failed to compile LangGraph: {compile_error}", exc_info=True)
        return None
