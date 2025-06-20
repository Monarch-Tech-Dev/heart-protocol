#!/bin/bash
# Heart Protocol Secret Rotation Script
# Safely rotate secrets in production

set -e

echo "🔄 Heart Protocol Secret Rotation"
echo "================================="

# Load current environment
if [ ! -f .env ]; then
    echo "❌ .env file not found. Run from Heart Protocol directory."
    exit 1
fi

source .env

echo "🔍 Current secrets status:"
echo "   SECRET_KEY: $(echo $SECRET_KEY | cut -c1-8)..."
echo "   DB_PASSWORD: $(echo $DB_PASSWORD | cut -c1-4)..."
echo "   JWT_SECRET: $(echo $JWT_SECRET | cut -c1-8)..."

echo ""
read -p "🤔 Which secrets do you want to rotate? (all/secret_key/db_password/jwt_secret): " rotation_choice

# Backup current .env
backup_file=".env.backup.$(date +%Y%m%d_%H%M%S)"
cp .env "$backup_file"
echo "✅ Backed up current .env to $backup_file"

case $rotation_choice in
    "all")
        echo "🔄 Rotating all secrets..."
        NEW_SECRET_KEY=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-32)
        NEW_DB_PASSWORD=$(openssl rand -base64 24 | tr -d "=+/" | cut -c1-24)
        NEW_JWT_SECRET=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-32)
        
        sed -i "s/SECRET_KEY=.*/SECRET_KEY=${NEW_SECRET_KEY}/" .env
        sed -i "s/DB_PASSWORD=.*/DB_PASSWORD=${NEW_DB_PASSWORD}/" .env
        sed -i "s/JWT_SECRET=.*/JWT_SECRET=${NEW_JWT_SECRET}/" .env
        echo "✅ Rotated: SECRET_KEY, DB_PASSWORD, JWT_SECRET"
        ;;
    "secret_key")
        NEW_SECRET_KEY=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-32)
        sed -i "s/SECRET_KEY=.*/SECRET_KEY=${NEW_SECRET_KEY}/" .env
        echo "✅ Rotated: SECRET_KEY"
        ;;
    "db_password")
        NEW_DB_PASSWORD=$(openssl rand -base64 24 | tr -d "=+/" | cut -c1-24)
        sed -i "s/DB_PASSWORD=.*/DB_PASSWORD=${NEW_DB_PASSWORD}/" .env
        echo "✅ Rotated: DB_PASSWORD"
        echo "⚠️  Database password rotation requires additional steps:"
        echo "   1. Update database user password"
        echo "   2. Restart all services"
        ;;
    "jwt_secret")
        NEW_JWT_SECRET=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-32)
        sed -i "s/JWT_SECRET=.*/JWT_SECRET=${NEW_JWT_SECRET}/" .env
        echo "✅ Rotated: JWT_SECRET"
        echo "⚠️  JWT rotation will invalidate all current user sessions"
        ;;
    *)
        echo "❌ Invalid choice. No secrets rotated."
        rm "$backup_file"
        exit 1
        ;;
esac

echo ""
echo "🔄 NEXT STEPS:"
echo "   1. Test configuration: docker-compose config"
echo "   2. Restart services: docker-compose restart"
echo "   3. Verify everything works"
echo "   4. Delete backup if successful: rm $backup_file"

echo ""
echo "📅 REMINDER: Set calendar reminder to rotate secrets quarterly"
echo "💚 Secret rotation complete! 🔐"