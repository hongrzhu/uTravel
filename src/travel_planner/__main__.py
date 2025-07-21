# """
# Main entry point for the travel planning application.
# """

# import logging
# from typing import Dict, Any, Optional
# from langchain_core.messages import HumanMessage, AIMessage
# from langgraph.graph import StateGraph, END

# from .core.agent import PlannerAgent
# from .config.settings import GEMINI_API_KEY, validate_api_keys

# class InteractivePlanState:
#     """State class for the interactive planning system."""
#     def __init__(self):
#         self.messages = []
#         self.current_plan = None
#         self.error_message = None

# def display_readable_plan(plan_json: Dict[str, Any]) -> None:
#     """Prints the plan in a user-friendly format."""
#     if not plan_json or not isinstance(plan_json.get("itinerary"), list):
#         print("No valid itinerary found to display.")
#         return

#     print("\n--- Current Itinerary ---")
#     itinerary_list = plan_json["itinerary"]
#     for day_plan in itinerary_list:
#         if not isinstance(day_plan, dict):
#             continue
            
#         print(f"\n** {day_plan.get('date', 'Unknown Date')} **")
#         if day_plan.get("daily_summary"):
#             print(f"   Summary: {day_plan['daily_summary']}")

#         activity_list = day_plan.get("activities", [])
#         if isinstance(activity_list, list):
#             for activity in activity_list:
#                 if not isinstance(activity, dict):
#                     continue
                    
#                 activity_name = activity.get('name', 'N/A')
#                 time_str = activity.get('time', '')
#                 time_display = f" ({time_str})" if time_str else ""
#                 print(f"- {activity_name}{time_display}")
                
#                 if activity.get('description'):
#                     print(f"    Desc: {activity['description']}")

#                 location_str = "N/A"
#                 top_level_address = activity.get('address')
#                 location_data = activity.get('location')
#                 if isinstance(top_level_address, str) and top_level_address:
#                     location_str = top_level_address
#                 elif isinstance(location_data, dict):
#                     nested_address = location_data.get('address')
#                     if isinstance(nested_address, str) and nested_address:
#                         location_str = nested_address
#                     else:
#                         lat = location_data.get('latitude', 'N/A')
#                         lon = location_data.get('longitude', 'N/A')
#                         if lat != 'N/A' or lon != 'N/A':
#                             location_str = f"Coords: (Lat: {lat}, Lon: {lon})"
#                 print(f"    Loc: {location_str}")

#                 if activity.get('budget'):
#                     print(f"    Budget: {activity['budget']}")
#                 if activity.get('notes'):
#                     print(f"    Notes: {activity['notes']}")
#         else:
#             print("   No activities found or 'activities' is not a list for this day.")

# def main():
#     """Main entry point for the travel planning application."""
#     print("--- Welcome to uTravel: Your Friendly AI Travel Companion! ---")
#     print("Tell me about your travel wishes! For example, 'I'd like a 3-day adventure in Paris focusing on museums and cafes.'")
#     print("Whenever you're ready to end our chat, just type 'exit' or 'quit.'")

#     if not validate_api_keys():
#         print("\nError: Missing required API keys. Please set the following environment variables:")
#         print("- GEMINI_API_KEY")
#         print("- MAPS_API_KEY")
#         print("- WEATHER_API_KEY")
#         return

#     # Initialize the planner agent
#     planner = PlannerAgent(GEMINI_API_KEY)
#     if not planner.llm:
#         print("\nError: Failed to initialize the AI system. Please check your API keys.")
#         return

#     # Initialize conversation state
#     state = InteractivePlanState()

#     while True:
#         try:
#             user_input = input("\nYou: ")
#             if user_input.lower() in ["exit", "quit"]:
#                 print("Thanks for chatting with uTravel. Safe travels!")
#                 break

#             # Add user message to history
#             state.messages.append(HumanMessage(content=user_input))
#             state.error_message = None

#             # Process the message
#             print("uTravel is crafting your journey...")
#             response = planner.process_message(state.messages)

#             # Update state
#             if response.get("messages"):
#                 state.messages.extend(response["messages"])
#             if response.get("error_message"):
#                 state.error_message = response["error_message"]

#             # Display response
#             last_ai_message = next((msg for msg in reversed(state.messages) if isinstance(msg, AIMessage)), None)
#             if last_ai_message:
#                 ai_content = last_ai_message.content
#                 print("\nuTravel:", end="")
#                 if isinstance(ai_content, list):
#                     for item in ai_content:
#                         if isinstance(item, str):
#                             if not (item.strip().startswith('{') or item.strip().startswith('```json')) or '"itinerary"' not in item:
#                                 print(f" {item}", end="")
#                     print()
#                 elif isinstance(ai_content, str):
#                     if not ((ai_content.strip().startswith('{') or ai_content.strip().startswith('```json')) and '"itinerary"' in ai_content):
#                         print(f" {ai_content}")
#                     else:
#                         print(" (Here's your personalized plan below!)")

#             # Display plan if available
#             if state.current_plan:
#                 display_readable_plan(state.current_plan)

#             # Display any errors
#             if state.error_message:
#                 print(f"\nHeads up: There was an issue while planning: {state.error_message}")

#         except KeyboardInterrupt:
#             print("\nThanks for visiting uTravel. Safe travels!")
#             break
#         except Exception as e:
#             logging.error(f"Unexpected error: {e}", exc_info=True)
#             print(f"\nSorry, an unexpected error occurred: {e}")

#     print("\n--- Thank you for using uTravel. Until next time! ---")

# if __name__ == "__main__":
#     main() 

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage, BaseMessage
from langchain_core.tools import tool

import googlemaps
from google.api_core import exceptions as google_exceptions
import requests
import json
import os
from datetime import datetime, timedelta, timezone
import pandas as pd
import logging
from typing import TypedDict, Annotated, List, Sequence, Optional, Dict, Any
import operator
import re # For potentially extracting info

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver # Keep if you want state saving

# --- 2. Configuration & Setup ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    GEMINI_API_KEY = user_secrets.get_secret("GEMINI_API_KEY")
    MAPS_API_KEY = user_secrets.get_secret("MAPS_API_KEY")
    WEATHER_API_KEY = user_secrets.get_secret("WEATHER_API_KEY") # Uncomment if using real weather API
    logging.info("Successfully retrieved API keys from Kaggle Secrets.")
except Exception as e:
    logging.warning(f"Could not retrieve keys from Kaggle Secrets (may be normal outside Kaggle): {e}")
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    MAPS_API_KEY = os.environ.get("MAPS_API_KEY")
    WEATHER_API_KEY = os.environ.get("WEATHER_API_KEY") # Uncomment if using real weather API

# --- 2. Configuration & Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

OWM_ONECALL_ENDPOINT = "https://api.openweathermap.org/data/3.0/onecall"
if not GEMINI_API_KEY:
    logging.error("GEMINI_API_KEY not found in Kaggle Secrets or environment variables.")
    # raise ValueError("Missing Gemini API Key")
    print("ERROR: GEMINI_API_KEY not found. LLM will not function.")
if not MAPS_API_KEY:
    logging.warning("MAPS_API_KEY not found. Google Maps features (Places, Directions) will fail or use dummy data.")
    # Set a dummy key to avoid crashing the googlemaps client init, but calls will fail
    MAPS_API_KEY = "YOUR_MAPS_API_KEY_HERE" # Placeholder

# Configure Google Maps Client
try:
    gmaps = googlemaps.Client(key=MAPS_API_KEY)
    if "YOUR_MAPS_API_KEY" in MAPS_API_KEY:
         logging.warning("Using placeholder Google Maps API key. Maps calls will fail.")
         gmaps_active = False
    else:
         gmaps_active = True
         logging.info("Google Maps client initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize Google Maps client: {e}")
    gmaps = None
    gmaps_active = False

# --- 3. Tool Definitions ---

def map_price_level(level: Optional[int]) -> str:
    """Maps Google Places price level (0-4) to $, $$, $$$ etc."""
    if level == 0: return "Free"
    if level == 1: return "$"
    if level == 2: return "$$"
    if level == 3: return "$$$"
    if level == 4: return "$$$$"


@tool
def get_weather_forecast(location: str, date: str) -> dict:
    """
    Gets the daily weather forecast for a specific location (city name) and date (YYYY-MM-DD)
    using the OpenWeatherMap One Call API. Requires latitude/longitude obtained via geocoding.
    Returns key forecast details like temperature, conditions, and precipitation probability.
    """
    logging.info(f"TOOL CALLED: get_weather_forecast(location='{location}', date='{date}')")

    if not WEATHER_API_KEY:
        logging.error("OpenWeatherMap API key is missing.")
        return {"error": "Weather API key not configured."}
    if not gmaps_active:
         logging.error("Google Maps client not active. Cannot geocode location for weather.")
         return {"error": "Maps service unavailable for geocoding."}

    # --- Step 1: Geocode location string to lat/lon ---
    lat, lon = None, None
    try:
        geocode_result = gmaps.geocode(location)
        if geocode_result and len(geocode_result) > 0:
            geometry = geocode_result[0].get('geometry', {})
            location_coords = geometry.get('location', {})
            lat = location_coords.get('lat')
            lon = location_coords.get('lng')
            logging.info(f"Geocoded '{location}' to Lat: {lat}, Lon: {lon}")
            if lat is None or lon is None:
                 raise ValueError("Geocoding result missing latitude or longitude.")
        else:
            logging.warning(f"Geocoding failed for location: {location}. Result: {geocode_result}")
            return {"error": f"Could not find coordinates for location: {location}"}
    except googlemaps.exceptions.ApiError as e:
        logging.error(f"Google Geocoding API error: {e}")
        return {"error": f"Geocoding API error: {e}"}
    except Exception as e:
        logging.error(f"Error during geocoding: {e}", exc_info=True)
        return {"error": f"An unexpected error occurred during geocoding: {e}"}

    # --- Step 2: Prepare and Call OWM API ---
    try:
        target_date_obj = datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError:
        logging.error(f"Invalid date format for weather forecast: {date}. Use YYYY-MM-DD.")
        return {"error": "Invalid date format. Please use YYYY-MM-DD."}

    params = {
        'lat': lat,
        'lon': lon,
        'appid': WEATHER_API_KEY,
        'units': 'metric', # Use Celsius
        'exclude': 'current,minutely,hourly,alerts' # We only need daily forecast
    }

    try:
        response = requests.get(OWM_ONECALL_ENDPOINT, params=params, timeout=10) # Add timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx, 5xx)
        weather_data = response.json()
        logging.info("Successfully received weather data from OWM.")
        # logging.debug(f"OWM Raw Response: {json.dumps(weather_data)[:500]}...") # Optional: log raw data snippet

    except requests.exceptions.Timeout:
         logging.error("OWM API request timed out.")
         return {"error": "Weather service request timed out."}
    except requests.exceptions.RequestException as e:
        logging.error(f"OWM API request failed: {e}")
        # Attempt to get error details from response if available
        error_detail = ""
        try:
            error_detail = response.json().get("message", "")
        except: # Ignore parsing errors if response wasn't JSON
             pass
        return {"error": f"Failed to fetch weather data: {e}. {error_detail}".strip()}
    except json.JSONDecodeError:
        logging.error("Failed to parse OWM API response as JSON.")
        return {"error": "Received invalid response from weather service."}

    # --- Step 3: Find the forecast for the target date ---
    daily_forecasts = weather_data.get('daily', [])
    if not daily_forecasts:
        logging.warning("No 'daily' forecast data found in OWM response.")
        return {"error": "No daily forecast data available from weather service."}

    target_forecast = None
    for day_forecast in daily_forecasts:
        dt_timestamp = day_forecast.get('dt')
        if dt_timestamp:
            # Convert timestamp to date object (UTC)
            forecast_date_obj = datetime.fromtimestamp(dt_timestamp, tz=timezone.utc).date()
            if forecast_date_obj == target_date_obj:
                target_forecast = day_forecast
                break # Found the date we need

    if not target_forecast:
        logging.warning(f"Forecast for the specific date {date} not found in the returned data (forecast may be too far out).")
        return {"error": f"Forecast for date {date} not available (max 8 days typical)."}

    # --- Step 4: Extract and format relevant data ---
    try:
        temp_info = target_forecast.get('temp', {})
        weather_info = target_forecast.get('weather', [{}])[0] # Get first weather condition
        precip_prob = target_forecast.get('pop', 0) * 100 # Probability of precipitation (0-1 -> 0-100)

        formatted_forecast = {
            "date": date,
            "location": location, # Return original location string for context
            "latitude": lat,    # Include coords used
            "longitude": lon,
            "temp_high_c": temp_info.get('max'),
            "temp_low_c": temp_info.get('min'),
            "conditions_main": weather_info.get('main', 'N/A'), # e.g., "Rain", "Clouds"
            "conditions_desc": weather_info.get('description', 'N/A'), # e.g., "light rain"
            "precip_prob_percent": round(precip_prob, 1), # Round percentage
            "summary": target_forecast.get('summary', '') # OWM provides a summary
        }
        logging.info(f"Extracted forecast for {date}: {formatted_forecast}")
        return formatted_forecast
    except Exception as e:
        logging.error(f"Error parsing extracted OWM forecast data: {e}", exc_info=True)
        return {"error": "Failed to parse weather data structure."}

@tool
def find_places_nearby(city: str, interests: List[str], keyword: Optional[str] = None, place_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Finds relevant places (attractions, restaurants, museums, etc.) in a city based on interests, specific keywords or place types using Google Places API.
    Provide general interests (e.g., ['art', 'history', 'cheap eats']).
    Optionally provide a specific keyword (e.g., 'Golden Gate Bridge') or a place_type (e.g., 'museum', 'restaurant', 'park').
    Returns a list of places with details like name, address, coordinates, rating, price level, and types.
    """
    logging.info(f"TOOL CALLED: find_places_nearby(city='{city}', interests={interests}, keyword='{keyword}', place_type='{place_type}')")
    if not gmaps_active:
        logging.error("Google Maps client not active. Cannot search for places.")
        # Return limited mock data if needed for testing flow without API key
        if "museum" in interests or keyword == "SFMOMA":
             return [{
                "place_id": "mock_sfmoma", "name": "SFMOMA (Mock)", "address": "151 3rd St, San Francisco, CA 94103",
                "latitude": 37.7857, "longitude": -122.4010, "rating": 4.5, "price_level_str": "$$",
                "types": ["museum", "art_gallery", "point_of_interest"], "status": "OPERATIONAL"
             }]
        return [{"error": "Maps service not available."}]

    # Construct query - prioritize keyword, then interests/type
    query = ""
    if keyword:
        query = f"{keyword} in {city}"
    elif interests:
        query = f"{' '.join(interests)} in {city}"
    elif place_type:
         query = f"{place_type} in {city}"
    else:
        return {"error": "Must provide interests, keyword, or place_type."}

    logging.info(f"Executing Google Places text search with query: '{query}'")
    try:
        # Use text search for more flexible querying
        places_result = gmaps.places(query=query) # Can also use places_nearby with location bias

        if places_result.get('status') == 'OK':
            results = []
            for place in places_result.get('results', []):
                location = place.get('geometry', {}).get('location', {})
                details = {
                    "place_id": place.get('place_id'),
                    "name": place.get('name'),
                    "address": place.get('formatted_address'),
                    "latitude": location.get('lat'),
                    "longitude": location.get('lng'),
                    "rating": place.get('rating'),
                    "user_ratings_total": place.get('user_ratings_total'),
                    "price_level_int": place.get('price_level'), # Keep original int if needed
                    "price_level_str": map_price_level(place.get('price_level')), # Add mapped string
                    "types": place.get('types'),
                    "status": place.get('business_status') # e.g., OPERATIONAL, CLOSED_TEMPORARILY
                }
                # Basic check if place seems relevant (can be improved)
                # E.g. check if place types match interests broadly
                is_relevant = True # Assume relevant for now, let LLM filter further
                if is_relevant:
                    results.append(details)

            logging.info(f"Found {len(results)} places matching query.")
            return results[:15] # Limit results
        else:
            logging.warning(f"Google Places API returned status: {places_result.get('status')}")
            return {"error": f"Places API error: {places_result.get('status')}", "details": places_result.get('error_message', '')}

    except googlemaps.exceptions.ApiError as e:
        logging.error(f"Google Places API error: {e}")
        return {"error": f"Maps API error: {e}"}
    except Exception as e:
        logging.error(f"Error finding places: {e}")
        return {"error": f"An unexpected error occurred: {e}"}


@tool
def get_travel_info(origin_lat: float, origin_lon: float, dest_lat: float, dest_lon: float, mode: str) -> dict:
    """
    Gets estimated travel time and distance between two points specified by latitude/longitude coordinates,
    using a specified travel mode ('driving', 'walking', 'transit', 'bicycling').
    Use this to check feasibility when sequencing activities.
    """
    logging.info(f"TOOL CALLED: get_travel_info(origin=({origin_lat},{origin_lon}), dest=({dest_lat},{dest_lon}), mode='{mode}')")
    if not gmaps_active:
         logging.error("Google Maps client not active. Cannot get travel info.")
         # Dummy response
         return {"origin": f"({origin_lat},{origin_lon})", "destination": f"({dest_lat},{dest_lon})", "mode": mode, "duration_text": "15 mins", "duration_seconds": 900, "distance_text": "2.1 km", "distance_meters": 2100, "status": "OK_DUMMY"}

    origin_coords = (origin_lat, origin_lon)
    destination_coords = (dest_lat, dest_lon)
    gmaps_mode = mode.lower()
    if gmaps_mode not in ['driving', 'walking', 'transit', 'bicycling']:
        logging.warning(f"Invalid travel mode '{mode}'. Defaulting to 'driving'.")
        gmaps_mode = 'driving'

    try:
        now = datetime.now()
        # Use departure_time for better transit estimates if relevant
        directions_result = gmaps.directions(origin_coords,
                                             destination_coords,
                                             mode=gmaps_mode,
                                             departure_time=now if gmaps_mode == 'transit' else None)

        if directions_result and len(directions_result) > 0:
            leg = directions_result[0]['legs'][0]
            duration = leg.get('duration', {}) # Use .get for safety
            distance = leg.get('distance', {})
            result = {
                "origin": f"({origin_lat},{origin_lon})",
                "destination": f"({dest_lat},{dest_lon})",
                "mode": gmaps_mode,
                "duration_text": duration.get('text', 'N/A'),
                "duration_seconds": duration.get('value'),
                "distance_text": distance.get('text', 'N/A'),
                "distance_meters": distance.get('value'),
                "status": "OK"
            }
            logging.info(f"Travel info retrieved: {result['duration_text']}, {result['distance_text']}")
            return result
        else:
            logging.warning(f"No directions found: {origin_coords} to {destination_coords} via {mode}")
            return {"error": "No route found", "status": directions_result[0].get('status', 'ZERO_RESULTS') if directions_result else "ZERO_RESULTS"}

    except googlemaps.exceptions.ApiError as e:
        logging.error(f"Google Maps Directions API error: {e}")
        return {"error": f"Maps API error: {e}", "status": "API_ERROR"}
    except Exception as e:
        logging.error(f"Error getting travel info: {e}")
        return {"error": f"An unexpected error occurred: {e}", "status": "REQUEST_FAILED"}

# List of all tools for LangGraph
tools = [get_weather_forecast, find_places_nearby, get_travel_info]

# --- 4. LangGraph State Definition ---

class InteractivePlanState(TypedDict):
    # Primary driver: the conversation history
    messages: Annotated[Sequence[BaseMessage], operator.add]

    # Stores the latest successfully parsed plan JSON
    current_plan: Optional[Dict[str, Any]]

    # User preferences extracted from conversation
    # target_city: Optional[str]
    # start_date_str: Optional[str]
    # end_date_str: Optional[str]
    # interests: Optional[List[str]]
    # budget_preference: Optional[str]
    # pace: Optional[str]
    # travel_mode: Optional[str]

    # Error tracking for the current turn
    error_message: Optional[str]

# --- 5. LangGraph Nodes ---

llm = None # Initialize as None
if not GEMINI_API_KEY or "YOUR_GEMINI_API_KEY" in GEMINI_API_KEY:
     logging.error("Gemini API Key is missing or invalid. Cannot initialize LLM.")
     print("ERROR: Gemini API Key is missing or invalid. LLM will not function.") # Added print statement
else:
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro", 
            temperature=0.7,
            # *** Explicitly pass the API key here ***
            google_api_key=GEMINI_API_KEY,
            safety_settings={ # Be careful adjusting these
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            },
            # convert_system_message_to_human=True
        )
        # Test connection (optional but recommended)
        # response = llm.invoke("Hello!")
        # logging.info(f"LLM connection test successful: {response.content[:50]}...")
        logging.info("ChatGoogleGenerativeAI model initialized successfully WITH explicit API KEY.")
    except Exception as e:
        logging.error(f"Failed to initialize ChatGoogleGenerativeAI: {e}", exc_info=True)
        # llm remains None
        print(f"ERROR: Failed to initialize LLM: {e}") # Added print statement

SYSTEM_PROMPT = """You are a helpful and conversational travel planning assistant. Your goal is to collaboratively create a personalized itinerary with the user.

**Your Responsibilities:**

1.  **Engage in Conversation:** Chat naturally with the user.
2.  **Gather Information:** If the user hasn't provided necessary details (target city, travel dates, interests, budget preference, preferred pace, travel mode), politely ask for them one by one until you have enough information to start planning.
3.  **Generate Initial Plan:** Once you have the core details, create a first draft of the itinerary in JSON format. Use the available tools (`get_weather_forecast`, `find_places_nearby`, `get_travel_info`) during this process to get required data (weather, place details including coordinates and address, travel times).
4.  **Use Tools Effectively:**
    *   Call `get_weather_forecast` for each day of the trip using the city name and date.
    *   Call `find_places_nearby` to find attractions, restaurants, etc., based on user interests or specific requests. Ensure you retrieve coordinates and address.
    *   Call `get_travel_info` using latitude and longitude coordinates from `find_places_nearby` results to calculate travel times between planned activities using the user's specified travel mode. Check feasibility based on pace.
5.  **Structure and Output JSON Plan:**
    *   When you have gathered enough information and are ready to present the initial plan OR present a revised plan based on feedback, **your primary output MUST be the complete itinerary formatted as a single JSON object.**
    *   This JSON object must have a root key "itinerary". The value can be a **list** of daily plan objects OR a **dictionary** where keys are "YYYY-MM-DD" dates and values are daily plan objects.
    *   Each daily plan object must contain a "date" (String) and "activities" (List).
    *   Each activity in the list must be an object containing ALL of the following fields, populated with data obtained from tools or reasoning:
        *   `name`: Specific place name (String)
        *   `time`: Estimated time range (String)
        *   `description`: Brief description (String)
        *   `location`: **Object** with `latitude` (Number) and `longitude` (Number).
        *   `address`: Formatted street address (String).
        *   `budget`: Estimated cost (String).
        *   `notes`: Relevant context (String, e.g., weather, calculated travel time). **No placeholders or tool notes.**
    *   **CRITICAL OUTPUT RULE:** When providing the plan, structure your response so the **JSON object is the main content**. You may add brief introductory or concluding conversational text *if necessary*, but the JSON structure itself MUST be present in your message content. Do not *only* provide conversational text when the user expects the plan.
6.  **Handle Feedback & Revise:** If the user provides feedback on the generated plan (e.g., "I don't like museums", "Can we swap Day 2 and Day 3?", "Add more cheap eats"), understand the request and generate a *revised* JSON itinerary incorporating their changes. Use tools again if needed for the revision.
7.  **Be Clear:** Explain your suggestions and incorporate user feedback transparently. If you cannot fulfill a request, explain why.

**Interaction Flow:** The user will provide input. You will process it based on the conversation history. You might ask clarifying questions, call tools, generate/revise the JSON plan, or provide conversational responses. Your final output for a turn should be either a question to the user, a request to call tools, or the complete JSON itinerary.
"""

# Node 1: Planner Agent (Handles conversation, planning, revision)
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
    # (or ensure it's passed correctly elsewhere)
    current_messages = list(messages)
    if not current_messages or not isinstance(current_messages[0], SystemMessage):
        current_messages.insert(0, SystemMessage(content=SYSTEM_PROMPT))

    llm_with_tools = llm.bind_tools(tools)

    try:
        # Invoke LLM with the full conversation history + system prompt
        ai_response = llm_with_tools.invoke(current_messages)
        logging.info(f"LLM Response: Type: {type(ai_response)}")
        # Log snippets
        if isinstance(ai_response.content, str):
            logging.info(f"LLM Response content snippet: {ai_response.content[:200]}...")
        if ai_response.tool_calls:
            logging.info(f"LLM Response tool calls: {ai_response.tool_calls}")

        # Return the AI response message to be added to the state
        # This will overwrite any previous error message for this turn
        return {"messages": [ai_response], "error_message": None}

    except google_exceptions.ResourceExhausted as e:
         logging.error(f"LLM API quota exceeded: {e}")
         # Return error message and a conversational AI message
         return {"messages": [AIMessage(content="Sorry, I encountered an API limit. Please try again later.")], "error_message": "Gemini API quota likely exceeded."}
    except Exception as e:
        logging.error(f"LLM invocation failed: {e}", exc_info=True)
        error_content = f"LLM Error: {e}"
        # Return error message and a conversational AI message
        return {"messages": [AIMessage(content=f"Sorry, an internal error occurred: {e}")], "error_message": error_content}

# Node 2: Tool Executor
def tool_executor_node(state: InteractivePlanState) -> Dict[str, Any]:
    """Executes tools called by the Planner Agent."""
    logging.info("--- Running Node: tool_executor_node ---")
    messages = list(state['messages']) # Get current messages
    last_message = messages[-1]

    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        logging.warning("Tool executor called, but last message has no tool calls.")
        return {}

    tool_calls = last_message.tool_calls
    logging.info(f"Executing {len(tool_calls)} tool calls: {[tc.get('name') for tc in tool_calls]}")

    tool_messages = []
    available_tools_map = {t.name: t for t in tools}

    for tool_call in tool_calls:
        tool_name = tool_call.get('name')
        tool_args = tool_call.get('args', {})
        tool_call_id = tool_call.get('id')

        if not tool_call_id:
             logging.error(f"Tool call missing 'id': {tool_call}")
             tool_messages.append(ToolMessage(content=json.dumps({"error": "Tool call missing 'id'."}), tool_call_id="error_missing_id_" + tool_name))
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
                     output_content = json.dumps({"error": f"Tool output serialization failed: {e}", "output_type": str(type(output))})
                tool_messages.append(ToolMessage(content=output_content, tool_call_id=tool_call_id))
            except Exception as e:
                logging.error(f"Error executing tool '{tool_name}': {e}", exc_info=True)
                tool_messages.append(ToolMessage(content=json.dumps({"error": f"Execution failed: {e}"}), tool_call_id=tool_call_id))
        else:
            logging.warning(f"LLM called unknown tool: '{tool_name}'")
            tool_messages.append(ToolMessage(content=json.dumps({"error": f"Unknown tool '{tool_name}' called."}), tool_call_id=tool_call_id))

    # Return ToolMessages and clear error from this turn
    return {"messages": tool_messages, "error_message": None}

# Node 3: Parse and Save Plan
def parse_and_save_plan_node(state: InteractivePlanState) -> Dict[str, Any]:
    """
    Parses potential JSON plan from the last AI message (which might be a string or list)
    and updates the state.
    """
    logging.info("--- Running Node: parse_and_save_plan_node ---")
    messages = state.get('messages', [])
    last_ai_message = messages[-1] if messages and isinstance(messages[-1], AIMessage) else None

    parsed_plan = None
    parsing_error = None
    content_to_parse = None

    if not last_ai_message or last_ai_message.tool_calls:
        logging.info("PARSE_DEBUG: Last message is not AI or has tool calls. No plan to parse.")
        return {} # Nothing to parse

    ai_content = last_ai_message.content
    logging.info(f"PARSE_DEBUG: AI Message Content Type: {type(ai_content)}")
    logging.info(f"PARSE_DEBUG: AI Message Content Raw: {ai_content}") # Log the raw content

    # --- Find the content containing the JSON ---
    if isinstance(ai_content, str):
        if '"itinerary"' in ai_content and (ai_content.strip().startswith('{') or ai_content.strip().startswith('```json')):
            content_to_parse = ai_content
            logging.info("PARSE_DEBUG: Identified potential JSON in string content.")
        else:
            logging.info("PARSE_DEBUG: String content does not appear to contain JSON plan.")
    elif isinstance(ai_content, list):
        logging.info("PARSE_DEBUG: AI content is a list. Searching for JSON string within it.")
        for item in ai_content:
            if isinstance(item, str) and '"itinerary"' in item and (item.strip().startswith('{') or item.strip().startswith('```json')):
                content_to_parse = item
                logging.info("PARSE_DEBUG: Found potential JSON string within the list.")
                break
        if not content_to_parse:
             logging.info("PARSE_DEBUG: No JSON string found within the list.")
    # ------------------------------------------

    if not content_to_parse:
        logging.info("PARSE_DEBUG: No content identified for parsing.")
        # Clear error state if it was just text
        if state.get("error_message"):
             return {"error_message": None}
        return {} # No plan found, return empty update

    # --- Try parsing the identified content ---
    logging.info(f"PARSE_DEBUG: Attempting to parse content: {content_to_parse[:500]}...") # Log snippet
    try:
        json_str = content_to_parse
        # Handle markdown code blocks
        if "```json" in json_str:
            logging.info("PARSE_DEBUG: Found JSON markdown block.")
            json_str = json_str.split("```json")[1].split("```")[0].strip()
        elif not json_str.strip().startswith('{'):
             logging.info("PARSE_DEBUG: Attempting fallback bracket finding.")
             start_index = json_str.find('{')
             end_index = json_str.rfind('}')
             if start_index != -1 and end_index != -1 and start_index < end_index:
                 json_str = json_str[start_index:end_index+1]
                 logging.info("PARSE_DEBUG: Extracted content using bracket finding.")
             else:
                 logging.error("PARSE_DEBUG: Could not find JSON object boundaries.")
                 raise json.JSONDecodeError("No clear JSON object boundaries found", json_str, 0)

        parsed_plan_candidate = json.loads(json_str)
        logging.info("PARSE_DEBUG: json.loads() successful.")

        # Basic validation
        if isinstance(parsed_plan_candidate, dict) and "itinerary" in parsed_plan_candidate and isinstance(parsed_plan_candidate["itinerary"], (dict, list)): # Allow dict or list
             logging.info(f"PARSE_DEBUG: Parsed JSON structure is valid (found 'itinerary' key with dict/list). Type: {type(parsed_plan_candidate['itinerary'])}")
             parsed_plan = parsed_plan_candidate
        else:
             logging.error("PARSE_DEBUG: Parsed JSON failed validation (missing 'itinerary' or wrong type).")
             parsed_plan = None
             parsing_error = "Parsed JSON has incorrect structure."

    except json.JSONDecodeError as e:
        logging.error(f"PARSE_DEBUG: json.loads() failed: {e}")
        parsed_plan = None
        parsing_error = f"Failed to parse final plan JSON: {e}"
    except Exception as e:
        logging.error(f"PARSE_DEBUG: Unexpected error during parsing: {e}", exc_info=True)
        parsed_plan = None
        parsing_error = f"Unexpected error parsing plan: {e}"
    # -----------------------------------------

    # Return update for state
    update = {"current_plan": parsed_plan}
    if parsing_error:
         update["error_message"] = parsing_error
         logging.error(f"PARSE_DEBUG: Setting error message: {parsing_error}")
    # Clear previous state error if we successfully parsed a plan now
    elif parsed_plan is not None and state.get("error_message"):
         update["error_message"] = None
         logging.info("PARSE_DEBUG: Clearing previous error message.")

    logging.info(f"PARSE_DEBUG: Returning update: {{'current_plan': {'set' if parsed_plan else 'None'}, 'error_message': {update.get('error_message')}}}")
    return update

# --- 6. LangGraph Edge Logic ---

def route_after_planner(state: InteractivePlanState) -> str:
    """Routes from the planner based on the AI's response."""
    messages = state['messages']
    last_message = messages[-1] if messages else None

    if isinstance(last_message, AIMessage):
        if last_message.tool_calls:
            logging.info("Conditional Edge: Routing to tool executor.")
            return "call_tools"
        else:
            # If no tool calls, assume it's text or a plan to be parsed
            logging.info("Conditional Edge: Routing to parse/save plan.")
            return "parse_plan"
    else:
        # Should not happen if planner ran correctly, route to end with error
        logging.warning(f"Conditional Edge: Unexpected message type after planner ({type(last_message)}). Routing to END.")
        return "error_end" # Or just END?

# --- 7. Define the Graph ---

workflow = StateGraph(InteractivePlanState)

# Add nodes
workflow.add_node("planner_agent", planner_agent_node)
workflow.add_node("tool_executor", tool_executor_node)
workflow.add_node("parse_and_save_plan", parse_and_save_plan_node)

# Define edges
workflow.set_entry_point("planner_agent") # Start with the planner

# Edge from tool executor back to planner
workflow.add_edge("tool_executor", "planner_agent")

# Conditional edge from planner
workflow.add_conditional_edges(
    "planner_agent",
    route_after_planner,
    {
        "call_tools": "tool_executor",
        "parse_plan": "parse_and_save_plan",
        "error_end": END # End graph if error occurs in planner or routing
    }
)

# Edge from parsing node to END (graph finishes its turn)
workflow.add_edge("parse_and_save_plan", END)

# Compile the graph
try:
    app = workflow.compile()
     # Optional: Visualize the graph
    from IPython.display import Image, display
    try:
        img_data = app.get_graph(xray=True).draw_mermaid_png()
        display(Image(img_data))
    except Exception as viz_error:
        print(f"Could not visualize graph: {viz_error}")
        
    logging.info("Interactive LangGraph compiled successfully.")
except Exception as compile_error:
    logging.error(f"Failed to compile LangGraph: {compile_error}", exc_info=True)
    app = None

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
    elif llm is None:  
        print("\nApologies! Our AI brains are taking a break. Please come back shortly.")  
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
