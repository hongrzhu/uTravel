"""
External API tools for the travel planning system.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import requests
import json
import googlemaps
from langchain_core.tools import tool

from ..config.settings import (
    OWM_ONECALL_ENDPOINT,
    WEATHER_API_KEY,
    MAPS_API_KEY
)

# Initialize Google Maps client
try:
    gmaps = googlemaps.Client(key=MAPS_API_KEY)
    gmaps_active = True
    logging.info("Google Maps client initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize Google Maps client: {e}")
    gmaps = None
    gmaps_active = False

def map_price_level(level: Optional[int]) -> str:
    """Maps Google Places price level (0-4) to $, $$, $$$ etc."""
    if level == 0: return "Free"
    if level == 1: return "$"
    if level == 2: return "$$"
    if level == 3: return "$$$"
    if level == 4: return "$$$$"
    return "Unknown"

@tool
def get_weather_forecast(location: str, date: str) -> dict:
    """Gets the daily weather forecast for a specific location and date."""
    logging.info(f"TOOL CALLED: get_weather_forecast(location='{location}', date='{date}')")

    if not WEATHER_API_KEY:
        return {"error": "Weather API key not configured."}
    if not gmaps_active:
        return {"error": "Maps service unavailable for geocoding."}

    # Geocode location
    try:
        geocode_result = gmaps.geocode(location)
        if not geocode_result:
            return {"error": f"Could not find coordinates for location: {location}"}
        
        lat = geocode_result[0]['geometry']['location']['lat']
        lon = geocode_result[0]['geometry']['location']['lng']
    except Exception as e:
        return {"error": f"Geocoding error: {str(e)}"}

    # Get weather data
    try:
        params = {
            'lat': lat,
            'lon': lon,
            'appid': WEATHER_API_KEY,
            'units': 'metric',
            'exclude': 'current,minutely,hourly,alerts'
        }
        response = requests.get(OWM_ONECALL_ENDPOINT, params=params, timeout=10)
        response.raise_for_status()
        weather_data = response.json()
    except Exception as e:
        return {"error": f"Weather API error: {str(e)}"}

    # Find forecast for target date
    target_date = datetime.strptime(date, "%Y-%m-%d").date()
    for day_forecast in weather_data.get('daily', []):
        forecast_date = datetime.fromtimestamp(day_forecast['dt'], tz=timezone.utc).date()
        if forecast_date == target_date:
            return {
                "date": date,
                "location": location,
                "latitude": lat,
                "longitude": lon,
                "temp_high_c": day_forecast['temp']['max'],
                "temp_low_c": day_forecast['temp']['min'],
                "conditions_main": day_forecast['weather'][0]['main'],
                "conditions_desc": day_forecast['weather'][0]['description'],
                "precip_prob_percent": round(day_forecast['pop'] * 100, 1),
                "summary": day_forecast.get('summary', '')
            }

    return {"error": f"No forecast available for {date}"}

@tool
def find_places_nearby(city: str, interests: List[str], keyword: Optional[str] = None, place_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """Finds relevant places in a city based on interests, keywords, or place types."""
    logging.info(f"TOOL CALLED: find_places_nearby(city='{city}', interests={interests}, keyword='{keyword}', place_type='{place_type}')")

    if not gmaps_active:
        return [{"error": "Maps service not available."}]

    # Construct query
    query = ""
    if keyword:
        query = f"{keyword} in {city}"
    elif interests:
        query = f"{' '.join(interests)} in {city}"
    elif place_type:
        query = f"{place_type} in {city}"
    else:
        return [{"error": "Must provide interests, keyword, or place_type."}]

    try:
        places_result = gmaps.places(query=query)
        if places_result.get('status') != 'OK':
            return [{"error": f"Places API error: {places_result.get('status')}"}]

        results = []
        for place in places_result.get('results', [])[:15]:  # Limit to 15 results
            location = place.get('geometry', {}).get('location', {})
            results.append({
                "place_id": place.get('place_id'),
                "name": place.get('name'),
                "address": place.get('formatted_address'),
                "latitude": location.get('lat'),
                "longitude": location.get('lng'),
                "rating": place.get('rating'),
                "user_ratings_total": place.get('user_ratings_total'),
                "price_level_str": map_price_level(place.get('price_level')),
                "types": place.get('types'),
                "status": place.get('business_status')
            })
        return results

    except Exception as e:
        return [{"error": f"Error finding places: {str(e)}"}]

@tool
def get_travel_info(origin_lat: float, origin_lon: float, dest_lat: float, dest_lon: float, mode: str) -> dict:
    """Gets estimated travel time and distance between two points."""
    logging.info(f"TOOL CALLED: get_travel_info(origin=({origin_lat},{origin_lon}), dest=({dest_lat},{dest_lon}), mode='{mode}')")

    if not gmaps_active:
        return {
            "origin": f"({origin_lat},{origin_lon})",
            "destination": f"({dest_lat},{dest_lon})",
            "mode": mode,
            "duration_text": "15 mins",
            "duration_seconds": 900,
            "distance_text": "2.1 km",
            "distance_meters": 2100,
            "status": "OK_DUMMY"
        }

    try:
        directions_result = gmaps.directions(
            (origin_lat, origin_lon),
            (dest_lat, dest_lon),
            mode=mode.lower(),
            departure_time=datetime.now() if mode.lower() == 'transit' else None
        )

        if not directions_result:
            return {"error": "No route found", "status": "ZERO_RESULTS"}

        leg = directions_result[0]['legs'][0]
        return {
            "origin": f"({origin_lat},{origin_lon})",
            "destination": f"({dest_lat},{dest_lon})",
            "mode": mode.lower(),
            "duration_text": leg['duration']['text'],
            "duration_seconds": leg['duration']['value'],
            "distance_text": leg['distance']['text'],
            "distance_meters": leg['distance']['value'],
            "status": "OK"
        }

    except Exception as e:
        return {"error": f"Error getting travel info: {str(e)}", "status": "REQUEST_FAILED"}

# List of all available tools
tools = [get_weather_forecast, find_places_nearby, get_travel_info] 