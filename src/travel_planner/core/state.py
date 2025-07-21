"""State definitions for the travel planning system."""

from typing import TypedDict, Annotated, Sequence, Optional, Dict, Any, List
import operator
from langchain_core.messages import BaseMessage

class InteractivePlanState(TypedDict):
    """State definition for the interactive travel planning system."""
    # Primary driver: the conversation history
    messages: Annotated[Sequence[BaseMessage], operator.add]

    # Stores the latest successfully parsed plan JSON
    current_plan: Optional[Dict[str, Any]]

    # Error tracking for the current turn
    error_message: Optional[str]

    # User preferences extracted from conversation (for future use)
    # target_city: Optional[str]
    # start_date_str: Optional[str]
    # end_date_str: Optional[str]
    # interests: Optional[List[str]]
    # budget_preference: Optional[str]
    # pace: Optional[str]
    # travel_mode: Optional[str]
