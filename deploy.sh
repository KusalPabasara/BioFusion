#!/bin/bash

# BioFusion Pneumonia Detection - Deployment Script
# VPS Deployment for 152.42.185.253

APP_NAME="biofusion-pneumonia"
APP_DIR="/var/www/$APP_NAME"
STREAMLIT_PORT=8502
PYTHON_VERSION="python3"

echo "=== BioFusion Pneumonia Detection - Deployment ==="

# Create app directory
sudo mkdir -p $APP_DIR
cd $APP_DIR

# Clone or update repository
if [ -d ".git" ]; then
    echo "Updating existing repository..."
    git pull origin main
else
    echo "Cloning repository..."
    git clone https://github.com/KusalPabasara/BioFusion.git .
fi

# Create virtual environment
echo "Setting up Python virtual environment..."
$PYTHON_VERSION -m venv venv
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r streamlit_app/requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Create systemd service file
echo "Creating systemd service..."
sudo tee /etc/systemd/system/$APP_NAME.service > /dev/null <<EOF
[Unit]
Description=BioFusion Pneumonia Detection Streamlit App
After=network.target

[Service]
Type=simple
User=www-data
Group=www-data
WorkingDirectory=$APP_DIR/streamlit_app
Environment="PATH=$APP_DIR/venv/bin"
ExecStart=$APP_DIR/venv/bin/streamlit run app.py --server.port $STREAMLIT_PORT --server.headless true --server.address 127.0.0.1
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# Set permissions
sudo chown -R www-data:www-data $APP_DIR

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable $APP_NAME
sudo systemctl restart $APP_NAME

echo "=== Deployment Complete ==="
echo "App is running on port $STREAMLIT_PORT"
echo "Configure nginx to proxy to http://127.0.0.1:$STREAMLIT_PORT"
