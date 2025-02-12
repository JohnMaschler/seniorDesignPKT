#!/bin/bash

# Step 1: Create a virtual environment
echo "Creating virtual environment..."
python3 -m venv env

# Step 2: Activate virtual environment
echo "Activating virtual environment..."
source env/bin/activate

# Step 3: Install dpkt
echo "Installing dpkt and others..."
pip install dpkt torch matplotlib

# Step 4: Verify installation
if python -c "import dpkt" 2>/dev/null; then
    echo "dpkt installed successfully!"
else
    echo "Failed to install dpkt. Try running 'pip install dpkt' manually inside the virtual environment."
fi


# Keep the virtual environment activated
echo "Virtual environment is now set up. Run 'source env/bin/activate' to use it."


