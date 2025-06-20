#!/bin/bash
# Heart Protocol Security Check Script
# Verifies security configuration before going live

set -e

echo "🔍 Heart Protocol Security Verification"
echo "======================================="

ISSUES=0

echo "🔐 Checking for exposed secrets..."

# Check if .env exists and is properly secured
if [ -f .env ]; then
    echo "✅ .env file exists"
    
    # Check permissions
    PERMS=$(stat -c "%a" .env)
    if [ "$PERMS" != "600" ]; then
        echo "❌ .env permissions are $PERMS (should be 600)"
        ISSUES=$((ISSUES + 1))
    else
        echo "✅ .env permissions are secure (600)"
    fi
else
    echo "❌ .env file missing"
    ISSUES=$((ISSUES + 1))
fi

# Check .gitignore
if grep -q ".env" .gitignore; then
    echo "✅ .env is in .gitignore"
else
    echo "❌ .env not in .gitignore - CRITICAL SECURITY ISSUE!"
    ISSUES=$((ISSUES + 1))
fi

# Check for secrets in git history
echo "🔍 Checking git history for exposed secrets..."
if git log --all --full-history -- .env | grep -q "commit"; then
    echo "❌ .env found in git history - secrets may be exposed!"
    echo "   Run: git filter-branch --force --index-filter 'git rm --cached --ignore-unmatch .env' --prune-empty --tag-name-filter cat -- --all"
    ISSUES=$((ISSUES + 1))
else
    echo "✅ No .env in git history"
fi

# Check for hardcoded secrets in code
echo "🔍 Scanning code for hardcoded secrets..."
SECRET_PATTERNS=("password" "secret" "key" "token" "api_key")
for pattern in "${SECRET_PATTERNS[@]}"; do
    if grep -r --include="*.py" --include="*.js" --include="*.yml" --exclude-dir=".git" -i "$pattern.*=" . | grep -v ".env" | grep -v "example" | grep -v "template" | grep -v "TODO" | grep -v "REPLACE"; then
        echo "⚠️  Found potential hardcoded secrets with pattern: $pattern"
        ISSUES=$((ISSUES + 1))
    fi
done

# Check Docker configuration
echo "🐳 Checking Docker security..."
if docker-compose config | grep -E "password|secret|key" | grep -v "\${"; then
    echo "❌ Hardcoded secrets found in Docker configuration!"
    ISSUES=$((ISSUES + 1))
else
    echo "✅ Docker configuration uses environment variables"
fi

# Check if services are using HTTPS
echo "🔒 Checking HTTPS configuration..."
if grep -q "CORS_ORIGINS=https://" .env 2>/dev/null; then
    echo "✅ CORS configured for HTTPS"
else
    echo "⚠️  CORS not configured for HTTPS"
fi

# Check for debug mode in production
echo "🐛 Checking debug settings..."
if grep -q "DEBUG=false" .env 2>/dev/null; then
    echo "✅ Debug mode disabled"
elif grep -q "DEBUG=true" .env 2>/dev/null; then
    echo "⚠️  Debug mode enabled - disable for production"
    ISSUES=$((ISSUES + 1))
fi

# Check secret strength
echo "🔑 Checking secret strength..."
if [ -f .env ]; then
    source .env
    
    if [ ${#SECRET_KEY} -lt 32 ]; then
        echo "❌ SECRET_KEY too short (${#SECRET_KEY} chars, need 32+)"
        ISSUES=$((ISSUES + 1))
    else
        echo "✅ SECRET_KEY length adequate"
    fi
    
    if [ ${#DB_PASSWORD} -lt 16 ]; then
        echo "❌ DB_PASSWORD too short (${#DB_PASSWORD} chars, need 16+)"
        ISSUES=$((ISSUES + 1))
    else
        echo "✅ DB_PASSWORD length adequate"
    fi
fi

# Check firewall status (if on Linux)
if command -v ufw &> /dev/null; then
    echo "🛡️  Checking firewall status..."
    if ufw status | grep -q "Status: active"; then
        echo "✅ UFW firewall is active"
    else
        echo "⚠️  UFW firewall not active"
    fi
fi

# Summary
echo ""
echo "📊 SECURITY AUDIT SUMMARY"
echo "========================="

if [ $ISSUES -eq 0 ]; then
    echo "✅ ALL SECURITY CHECKS PASSED!"
    echo "🚀 Heart Protocol is ready for secure deployment"
    exit 0
else
    echo "❌ FOUND $ISSUES SECURITY ISSUES"
    echo "🛑 PLEASE FIX ISSUES BEFORE DEPLOYMENT"
    echo ""
    echo "🔧 Common fixes:"
    echo "   - chmod 600 .env"
    echo "   - Add .env to .gitignore"
    echo "   - Remove secrets from git history"
    echo "   - Use environment variables instead of hardcoded values"
    echo "   - Generate stronger passwords"
    echo "   - Enable firewall: sudo ufw enable"
    exit 1
fi