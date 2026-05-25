#!/bin/bash

set -e

# Go to project root
cd "$(dirname "$0")"

# Create virtual environment if missing
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3.11 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install/update dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Run application
echo "Starting webcam application..."
python3.11 main.py