# 🌍 uTravel: AI-Powered Travel Planning Assistant

uTravel is an intelligent travel planning assistant that helps users create personalized travel itineraries through natural conversation. Powered by Google's Gemini Pro LLM and integrated with real-time data from Google Maps and OpenWeatherMap APIs, uTravel provides dynamic, context-aware travel planning.

## Features

- **Natural Conversation Interface**: Interact with the AI assistant using natural language
- **Real-time Data Integration**:
  - Weather forecasts for planned travel dates
  - Location and attraction information from Google Maps
  - Travel time and distance calculations
- **Personalized Planning**:
  - Adapts to user preferences and interests
  - Considers budget constraints
  - Accounts for weather conditions
  - Optimizes for preferred pace and transportation mode
- **Interactive Refinement**: Modify and refine plans through conversation
- **Structured Output**: Generates well-organized, detailed itineraries

## Getting Started

### Prerequisites

- Python 3.8 or higher
- API keys for:
  - Google Gemini Pro
  - Google Maps Platform
  - OpenWeatherMap

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/hongrzhu/uTravel.git
   cd uTravel
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   export GEMINI_API_KEY="your_gemini_api_key"
   export MAPS_API_KEY="your_google_maps_api_key"
   export WEATHER_API_KEY="your_openweathermap_api_key"
   ```

### Usage

Run the application:
```bash
python -m travel_planner
```

Example interaction:
```
You: I'd like a 3-day trip to Paris focusing on museums and cafes, mid-range budget, moderate pace, using public transport. Dates: 2024-05-01 to 2024-05-03.

uTravel: I'll help you plan your Paris adventure! Let me gather some information about attractions and check the weather forecast...
```

## Project Structure

```
uTravel/
├── src/
│   └── travel_planner/
│       ├── core/
│       │   ├── __init__.py
│       │   └── agent.py          # Main planner agent implementation
│       ├── utils/
│       │   ├── __init__.py
│       │   └── tools.py          # External API integrations
│       ├── config/
│       │   ├── __init__.py
│       │   └── settings.py       # Configuration and constants
│       └── __main__.py           # Application entry point
├── frontend/                     # Future web interface development
├── backend/                      # Future backend service development
├── tests/                        # Test directory
├── docs/                         # Documentation
├── requirements.txt              # Project dependencies
├── setup.py                      # Package installation configuration
└── README.md                     # This file
```

## Technical Details

- **LLM Integration**: Uses Google's Gemini Pro model through LangChain
- **State Management**: Implements a state-based conversation system
- **API Integration**:
  - Google Maps Platform (Places, Directions, Geocoding)
  - OpenWeatherMap (Weather forecasts)
- **Error Handling**: Robust error handling and graceful degradation
- **Logging**: Comprehensive logging system for debugging

## Future Improvements
- [ ] Agent workflow optimization
- [ ] Web-based user interface
- [ ] User profile and preference storage
- [ ] Integration with booking services
- [ ] Multi-language support
- [ ] Mobile application
- [ ] Offline mode with cached data
- [ ] Social sharing features
- [ ] Trip cost estimation
- [ ] Real-time updates for weather and traffic

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Google Gemini Pro for the LLM capabilities
- Google Maps Platform for location services
- OpenWeatherMap for weather data
- LangChain for the AI framework
- All contributors and users of the project 