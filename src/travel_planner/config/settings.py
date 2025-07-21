"""
Configuration settings and constants for the travel planning system.
"""

import os
import logging
from typing import Optional
from langchain_google_genai import HarmBlockThreshold, HarmCategory
import googlemaps
from google.api_core import exceptions as google_exceptions

try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    GEMINI_API_KEY = user_secrets.get_secret("GEMINI_API_KEY")
    MAPS_API_KEY = user_secrets.get_secret("MAPS_API_KEY")
    WEATHER_API_KEY = user_secrets.get_secret("WEATHER_API_KEY")
    logging.info("Successfully retrieved API keys from Kaggle Secrets.")
except Exception as e:
    logging.warning(f"Could not retrieve keys from Kaggle Secrets (may be normal outside Kaggle): {e}")
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    MAPS_API_KEY = os.environ.get("MAPS_API_KEY")
    WEATHER_API_KEY = os.environ.get("WEATHER_API_KEY")

# API Keys
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MAPS_API_KEY = os.environ.get("MAPS_API_KEY")
WEATHER_API_KEY = os.environ.get("WEATHER_API_KEY")

# API Endpoints
OWM_ONECALL_ENDPOINT = "https://api.openweathermap.org/data/3.0/onecall"

# System Prompt
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
6.  **Handle Feedback & Revise:** If the user provides feedback on the generated plan (e.g., "I don't like museums", "Can we swap Day 2 and Day 3?", "Add more cheap eats"), understand the request and generate a *revised* JSON itinerary incorporating their changes. Use tools again if needed for the revision.
7.  **Be Clear:** Explain your suggestions and incorporate user feedback transparently. If you cannot fulfill a request, explain why.
"""

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)

def validate_api_keys() -> bool:
    """Validate that all required API keys are present."""
    missing_keys = []
    
    if not GEMINI_API_KEY:
        missing_keys.append("GEMINI_API_KEY")
        logging.error("GEMINI_API_KEY not found. LLM will not function.")
    if not MAPS_API_KEY:
        missing_keys.append("MAPS_API_KEY")
        logging.warning("MAPS_API_KEY not found. Google Maps features will fail or use dummy data.")
    if not WEATHER_API_KEY:
        missing_keys.append("WEATHER_API_KEY")
        
    if missing_keys:
        logging.error(f"Missing required API keys: {', '.join(missing_keys)}")
        return False
        
    return True

# --- Google Maps Client Setup ---
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

# --- LLM Configuration ---
GEMINI_MODEL_CONFIG = {
    "model": "gemini-2.5-pro",
    "temperature": 0.7,
    "safety_settings": {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }
}