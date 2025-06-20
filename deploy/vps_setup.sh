#!/bin/bash
# Heart Protocol VPS Deployment Script
# Deploy Monarch Bot to a simple VPS (DigitalOcean, Linode, etc.)

echo "🦋 Heart Protocol VPS Deployment"
echo "💙 Setting up Monarch Bot for 24/7 operation"

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install -y python3 python3-pip python3-venv git

# Clone repository (if not already there)
if [ ! -d "Heart-Protocol" ]; then
    git clone https://github.com/your-username/Heart-Protocol.git
fi

cd Heart-Protocol

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create systemd service for 24/7 operation
sudo tee /etc/systemd/system/monarch-bot.service > /dev/null <<EOF
[Unit]
Description=Monarch Bot - Heart Protocol
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/home/$USER/Heart-Protocol
Environment=PATH=/home/$USER/Heart-Protocol/venv/bin
ExecStart=/home/$USER/Heart-Protocol/venv/bin/python launch_monarch.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable monarch-bot
sudo systemctl start monarch-bot

echo "✅ Monarch Bot deployed and running!"
echo "📊 Check status: sudo systemctl status monarch-bot"
echo "📝 View logs: sudo journalctl -u monarch-bot -f"