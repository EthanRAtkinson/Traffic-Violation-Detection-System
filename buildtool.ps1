# This script creates a virtual environment and installs dependencies.

py -3 -m venv build_tool_env

# Activate the virtual environment
. .\build_tool_env\Scripts\Activate.ps1

# Upgrade pip and install required packages
#pip3 install --upgrade pip
pip3 install -r requirements.txt

Write-Host "Setup complete!"

# This script activates the virtual environment and runs app.py.
. .\build_tool_env\Scripts\Activate.ps1
py -3 app.py

# This script removes the virtual environment folder.
deactivate
Remove-Item -Recurse -Force .\build_tool_env
