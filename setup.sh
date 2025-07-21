#!/bin/bash
# This script sets up the environment for the travel planner application
#!/bin/bash

# Update system and install dependencies
sudo apt update
sudo apt install -y python3 python3-pip python3-venv

# Create a virtual environment (inside the current directory or specify a path)
VENV_DIR="myenv"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo "Virtual environment created at $VENV_DIR"
else
    echo "Virtual environment already exists at $VENV_DIR"
fi

# Activate the virtual environment
source "$VENV_DIR/bin/activate"

# Install required Python packages
echo "Installing required packages..."
pip install -q -r requirements.txt || {
    echo "Failed to install from requirements.txt, installing packages individually."
    pip install -q langgraph langchain googlemaps requests pandas ipython langchain-google-genai google-generativeai
}

# Deactivate the virtual environment after setup
deactivate

echo "Setup complete. Use 'source $VENV_DIR/bin/activate' to activate the virtual environment."
