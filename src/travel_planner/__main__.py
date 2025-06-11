"""
Main entry point for the travel planning application.
"""

import logging
from typing import Dict, Any, Optional
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

from .core.agent import PlannerAgent
from .config.settings import GEMINI_API_KEY, validate_api_keys

class InteractivePlanState:
    """State class for the interactive planning system."""
    def __init__(self):
        self.messages = []
        self.current_plan = None
        self.error_message = None

def display_readable_plan(plan_json: Dict[str, Any]) -> None:
    """Prints the plan in a user-friendly format."""
    if not plan_json or not isinstance(plan_json.get("itinerary"), list):
        print("No valid itinerary found to display.")
        return

    print("\n--- Current Itinerary ---")
    itinerary_list = plan_json["itinerary"]
    for day_plan in itinerary_list:
        if not isinstance(day_plan, dict):
            continue
            
        print(f"\n** {day_plan.get('date', 'Unknown Date')} **")
        if day_plan.get("daily_summary"):
            print(f"   Summary: {day_plan['daily_summary']}")

        activity_list = day_plan.get("activities", [])
        if isinstance(activity_list, list):
            for activity in activity_list:
                if not isinstance(activity, dict):
                    continue
                    
                activity_name = activity.get('name', 'N/A')
                time_str = activity.get('time', '')
                time_display = f" ({time_str})" if time_str else ""
                print(f"- {activity_name}{time_display}")
                
                if activity.get('description'):
                    print(f"    Desc: {activity['description']}")

                location_str = "N/A"
                top_level_address = activity.get('address')
                location_data = activity.get('location')
                if isinstance(top_level_address, str) and top_level_address:
                    location_str = top_level_address
                elif isinstance(location_data, dict):
                    nested_address = location_data.get('address')
                    if isinstance(nested_address, str) and nested_address:
                        location_str = nested_address
                    else:
                        lat = location_data.get('latitude', 'N/A')
                        lon = location_data.get('longitude', 'N/A')
                        if lat != 'N/A' or lon != 'N/A':
                            location_str = f"Coords: (Lat: {lat}, Lon: {lon})"
                print(f"    Loc: {location_str}")

                if activity.get('budget'):
                    print(f"    Budget: {activity['budget']}")
                if activity.get('notes'):
                    print(f"    Notes: {activity['notes']}")
        else:
            print("   No activities found or 'activities' is not a list for this day.")

def main():
    """Main entry point for the travel planning application."""
    print("--- Welcome to uTravel: Your Friendly AI Travel Companion! ---")
    print("Tell me about your travel wishes! For example, 'I'd like a 3-day adventure in Paris focusing on museums and cafes.'")
    print("Whenever you're ready to end our chat, just type 'exit' or 'quit.'")

    if not validate_api_keys():
        print("\nError: Missing required API keys. Please set the following environment variables:")
        print("- GEMINI_API_KEY")
        print("- MAPS_API_KEY")
        print("- WEATHER_API_KEY")
        return

    # Initialize the planner agent
    planner = PlannerAgent(GEMINI_API_KEY)
    if not planner.llm:
        print("\nError: Failed to initialize the AI system. Please check your API keys.")
        return

    # Initialize conversation state
    state = InteractivePlanState()

    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Thanks for chatting with uTravel. Safe travels!")
                break

            # Add user message to history
            state.messages.append(HumanMessage(content=user_input))
            state.error_message = None

            # Process the message
            print("uTravel is crafting your journey...")
            response = planner.process_message(state.messages)

            # Update state
            if response.get("messages"):
                state.messages.extend(response["messages"])
            if response.get("error_message"):
                state.error_message = response["error_message"]

            # Display response
            last_ai_message = next((msg for msg in reversed(state.messages) if isinstance(msg, AIMessage)), None)
            if last_ai_message:
                ai_content = last_ai_message.content
                print("\nuTravel:", end="")
                if isinstance(ai_content, list):
                    for item in ai_content:
                        if isinstance(item, str):
                            if not (item.strip().startswith('{') or item.strip().startswith('```json')) or '"itinerary"' not in item:
                                print(f" {item}", end="")
                    print()
                elif isinstance(ai_content, str):
                    if not ((ai_content.strip().startswith('{') or ai_content.strip().startswith('```json')) and '"itinerary"' in ai_content):
                        print(f" {ai_content}")
                    else:
                        print(" (Here's your personalized plan below!)")

            # Display plan if available
            if state.current_plan:
                display_readable_plan(state.current_plan)

            # Display any errors
            if state.error_message:
                print(f"\nHeads up: There was an issue while planning: {state.error_message}")

        except KeyboardInterrupt:
            print("\nThanks for visiting uTravel. Safe travels!")
            break
        except Exception as e:
            logging.error(f"Unexpected error: {e}", exc_info=True)
            print(f"\nSorry, an unexpected error occurred: {e}")

    print("\n--- Thank you for using uTravel. Until next time! ---")

if __name__ == "__main__":
    main() 