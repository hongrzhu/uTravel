
from travel_planner.core.graph import compile_graph
from langchain_core.messages import HumanMessage, AIMessage
import json
import logging

# Initialize the graph
app = compile_graph()

# --- Helper Function to Display Plan ---
def display_readable_plan(plan_json):
    """Prints the plan in a user-friendly format."""
    if not plan_json or not isinstance(plan_json.get("itinerary"), list):
        print("No valid itinerary found to display.")
        return

    print("\n--- Current Itinerary ---")
    itinerary_list = plan_json["itinerary"]
    for day_plan in itinerary_list:
        if not isinstance(day_plan, dict): continue
        print(f"\n** {day_plan.get('date', 'Unknown Date')} **")
        if day_plan.get("daily_summary"): print(f"   Summary: {day_plan['daily_summary']}")

        activity_list = day_plan.get("activities", [])
        if isinstance(activity_list, list):
            for activity in activity_list:
                 if not isinstance(activity, dict): continue
                 activity_name = activity.get('name', 'N/A')
                 time_str = activity.get('time', '')
                 time_display = f" ({time_str})" if time_str else ""
                 print(f"- {activity_name}{time_display}")
                 if activity.get('description'): print(f"    Desc: {activity['description']}")

                 # Prioritize top-level address, then nested, then coords
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
                         if lat != 'N/A' or lon != 'N/A': location_str = f"Coords: (Lat: {lat}, Lon: {lon})"
                 print(f"    Loc: {location_str}")

                 if activity.get('budget'): print(f"    Budget: {activity['budget']}")
                 if activity.get('notes'): print(f"    Notes: {activity['notes']}")
        else:
            print("   No activities found or 'activities' is not a list for this day.")

def main():
    print("--- Welcome to uTravel: Your Friendly AI Travel Companion! ---")  
    print("Tell me about your travel wishes! For example, 'I'd like a 3-day adventure in Paris focusing on museums and cafes.'")  
    print("Whenever you're ready to end our chat, just type 'exit' or 'quit.'")  
  
    if not app:  
        print("\nOops! It seems there's a hiccup with our planning system. Please try again later.")  
    else:  
        # Initialize conversation state  
        conversation_state = {  
            "messages": [],  
            "current_plan": None,  
            "error_message": None  
        }  
  
        while True:  
            try:  
                user_input = input("\nYou: ")  
                if user_input.lower() in ["exit", "quit"]:  
                    print("Thanks for chatting with uTravel. Safe travels!")  
                    break  
  
                # Add user message to history  
                conversation_state["messages"].append(HumanMessage(content=user_input))  
                conversation_state["error_message"] = None  
  
                # Optional initial info extraction logic here...  
  
                # --- Run the Graph for one turn ---  
                print("uTravel is crafting your journey...")  
                graph_output_state = None  
                try:  
                    current_graph_input = {  
                        "messages": conversation_state["messages"],  
                        "current_plan": conversation_state["current_plan"],  
                        "error_message": conversation_state["error_message"]  
                    }  
                    config = {"recursion_limit": 25}  
                    graph_output_state = app.invoke(current_graph_input, config=config)  
  
                except Exception as graph_run_error:  
                    logging.error(f"uTravel ran into an issue: {graph_run_error}", exc_info=True)  
                    print(f"\nSorry, there was a snag while planning your itinerary: {graph_run_error}")  
                    conversation_state["messages"].append(AIMessage(content="I'm sorry, I encountered an issue while processing your request."))  
  
                # --- Process Graph Output ---  
                if graph_output_state:  
                    conversation_state["messages"] = graph_output_state.get("messages", conversation_state["messages"])  
                    conversation_state["current_plan"] = graph_output_state.get("current_plan", conversation_state["current_plan"])  
                    conversation_state["error_message"] = graph_output_state.get("error_message", conversation_state["error_message"])  
  
                    last_ai_message = next((msg for msg in reversed(conversation_state["messages"]) if isinstance(msg, AIMessage)), None)  
                    ai_printed_response = False  
                    if last_ai_message:  
                        ai_content = last_ai_message.content  
                        print("\nuTravel:", end="")  
                        if isinstance(ai_content, list):  
                            for item in ai_content:  
                                if isinstance(item, str):  
                                    if not (item.strip().startswith('{') or item.strip().startswith('```json')) or '"itinerary"' not in item:  
                                        print(f" {item}", end="")  
                            print()  
                            ai_printed_response = True  
                        elif isinstance(ai_content, str):  
                             if not ((ai_content.strip().startswith('{') or ai_content.strip().startswith('```json')) and '"itinerary"' in ai_content):  
                                 print(f" {ai_content}")  
                                 ai_printed_response = True  
                             else:  
                                  print(" (Here's your personalized plan below!)")  
                                  ai_printed_response = True  
  
                    if not ai_printed_response:  
                         print(" (I couldn't generate a response this time. Let's give it another try!)")  
  
                    if conversation_state["current_plan"]:  
                         if last_ai_message and isinstance(last_ai_message.content, str) and '"itinerary"' in last_ai_message.content:  
                              display_readable_plan(conversation_state["current_plan"])  
                         elif not last_ai_message.tool_calls:  
                             display_readable_plan(conversation_state["current_plan"])  
  
                    if conversation_state["error_message"]:  
                         print(f"\nHeads up: There was an issue while planning: {conversation_state['error_message']}")  
  
                else:  
                     print("\nOh no! There was an issue with the planning process. Please try again.")  
  
            except EOFError:  
                print("\nThanks for visiting uTravel. Safe travels!")  
                break  
            except KeyboardInterrupt:  
                 print("\nThanks for visiting uTravel. Safe travels!")  
                 break  
  
    print("\n--- Thank you for using uTravel. Until next time! ---") 
    
if __name__ == "__main__":
    main()
