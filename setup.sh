#!/bin/bash

# Exit on error
set -e

echo "Starting environment setup..."

# Determine python command
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
elif command -v python &> /dev/null; then
    PYTHON_CMD=python
else
    echo "Error: Python is not installed."
    exit 1
fi

echo "Using Python command: $PYTHON_CMD"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment (.venv)..."
    $PYTHON_CMD -m venv .venv
else
    echo "Virtual environment (.venv) already exists."
fi

# Activate virtual environment
# Check for Windows-style Scripts folder or Unix-style bin folder
if [ -f ".venv/Scripts/activate" ]; then
    echo "Activating virtual environment (Windows style)..."
    source .venv/Scripts/activate
elif [ -f ".venv/bin/activate" ]; then
    echo "Activating virtual environment (Unix style)..."
    source .venv/bin/activate
else
    echo "Error: Could not find activation script in .venv"
    exit 1
fi

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "Warning: requirements.txt not found. Skipping dependency installation."
fi

# Run initialization script
if [ -f "initialize.py" ]; then
    echo "Running initialization script (Download data & Precompute cache)..."
    python initialize.py
else
    echo "Error: initialize.py not found."
    exit 1
fi

echo "----------------------------------------------------------------"
echo "Setup complete!"
echo ""
echo "To activate the environment in your current shell, run:"
if [ -f ".venv/Scripts/activate" ]; then
    echo "  source .venv/Scripts/activate"
elif [ -f ".venv/bin/activate" ]; then
    echo "  source .venv/bin/activate"
fi
echo "----------------------------------------------------------------"
