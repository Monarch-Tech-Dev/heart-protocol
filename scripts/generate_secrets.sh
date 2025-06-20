#!/bin/bash
# Heart Protocol Secret Generation Script
# Generates secure secrets for production deployment

set -e

echo "🔐 Heart Protocol Secret Generator"
echo "=================================="

# Check if openssl is available
if ! command -v openssl &> /dev/null; then
    echo "❌ OpenSSL is required but not installed."
    echo "   Install with: sudo apt install openssl"
    exit 1
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        cp .env.example .env
        echo "✅ Created .env from .env.example"
    else
        echo "❌ .env.example not found. Please create it first."
        exit 1
    fi
else
    echo "⚠️  .env file already exists. Creating backup..."
    cp .env .env.backup.$(date +%Y%m%d_%H%M%S)
fi

echo ""
echo "🔑 Generating secure secrets..."

# Generate secrets
SECRET_KEY=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-32)
DB_PASSWORD=$(openssl rand -base64 24 | tr -d "=+/" | cut -c1-24)
ENCRYPTION_KEY=$(openssl rand -hex 32)
JWT_SECRET=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-32)
WEBHOOK_SECRET=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-32)
GRAFANA_PASSWORD=$(openssl rand -base64 16 | tr -d "=+/" | cut -c1-16)

# Update .env file with generated secrets
sed -i "s/SECRET_KEY=.*/SECRET_KEY=${SECRET_KEY}/" .env
sed -i "s/DB_PASSWORD=.*/DB_PASSWORD=${DB_PASSWORD}/" .env
sed -i "s/ENCRYPTION_KEY=.*/ENCRYPTION_KEY=${ENCRYPTION_KEY}/" .env
sed -i "s/JWT_SECRET=.*/JWT_SECRET=${JWT_SECRET}/" .env
sed -i "s/WEBHOOK_SECRET=.*/WEBHOOK_SECRET=${WEBHOOK_SECRET}/" .env
sed -i "s/GRAFANA_PASSWORD=.*/GRAFANA_PASSWORD=${GRAFANA_PASSWORD}/" .env

echo "✅ Generated and updated the following secrets:"
echo "   - SECRET_KEY (32 characters)"
echo "   - DB_PASSWORD (24 characters)"
echo "   - ENCRYPTION_KEY (64 hex characters)"
echo "   - JWT_SECRET (32 characters)"
echo "   - WEBHOOK_SECRET (32 characters)"
echo "   - GRAFANA_PASSWORD (16 characters)"

echo ""
echo "🔒 SECURITY REMINDERS:"
echo "   1. NEVER commit .env to git!"
echo "   2. Set proper file permissions: chmod 600 .env"
echo "   3. Store backups securely"
echo "   4. Rotate secrets regularly"

echo ""
echo "📝 YOU STILL NEED TO MANUALLY ADD:"
echo "   - BLUESKY_HANDLE=your-bot-handle.bsky.social"
echo "   - BLUESKY_APP_PASSWORD=your-bluesky-app-password"
echo "   - OPENAI_API_KEY=your-openai-key (if using AI features)"
echo "   - SENTRY_DSN=your-sentry-dsn (if using error tracking)"
echo "   - CORS_ORIGINS=https://your-domain.com"

echo ""
echo "🛠️  NEXT STEPS:"
echo "   1. Edit .env and add your Bluesky credentials"
echo "   2. Set secure permissions: chmod 600 .env"
echo "   3. Test your configuration: docker-compose config"
echo "   4. Deploy: docker-compose up -d"

echo ""
echo "💚 Secrets generated successfully! Keep them safe! 🔐"