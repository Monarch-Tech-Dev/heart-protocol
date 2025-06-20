# 🚀 Heart Protocol Deployment Guide

## 🌟 **Quick Start**

### **Prerequisites**
- Docker & Docker Compose
- Node.js 18+ (for development)
- Python 3.9+ (for development)
- Git

### **One-Command Deployment**
```bash
# Clone and deploy in one step
git clone https://github.com/your-org/heart-protocol.git
cd heart-protocol
cp .env.example .env
# Edit .env with your configuration
docker-compose up -d
```

---

## 🏗️ **Production Deployment**

### **1. Environment Setup**

**Copy and configure environment:**
```bash
cp .env.example .env
```

**Required Environment Variables:**
```bash
# === CRITICAL - CHANGE THESE ===
SECRET_KEY=your-super-secret-key-here-change-this
DB_PASSWORD=your-secure-database-password
BLUESKY_APP_PASSWORD=your-bluesky-app-password
GRAFANA_PASSWORD=your-grafana-admin-password

# === BLUESKY INTEGRATION ===
BLUESKY_HANDLE=monarch.bsky.social
MONARCH_BLUESKY_HANDLE=monarch.bsky.social

# === SECURITY ===
ENVIRONMENT=production
DEBUG=false
CORS_ORIGINS=https://your-domain.com
```

### **2. Docker Deployment**

**Start all services:**
```bash
# Production deployment
docker-compose up -d

# Check service health
docker-compose ps
docker-compose logs heart-protocol-api
```

**Services included:**
- **Heart Protocol API** (Port 8000) - Core caring algorithms
- **Monarch Bot** - Bluesky interaction service  
- **Care Monitor** - Real-time firehose monitoring
- **PostgreSQL** (Port 5432) - Primary database
- **Redis** (Port 6379) - Caching and sessions
- **Prometheus** (Port 9090) - Metrics collection
- **Grafana** (Port 3000) - Monitoring dashboard
- **Nginx** (Ports 80/443) - Reverse proxy and SSL

### **3. SSL Certificate Setup**

**Using Let's Encrypt (Recommended):**
```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Get certificates
sudo certbot --nginx -d your-domain.com -d api.your-domain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

### **4. Database Setup**

**Initial database setup:**
```bash
# Run database migrations
docker-compose exec heart-protocol-api python -m alembic upgrade head

# Create initial admin user (optional)
docker-compose exec heart-protocol-api python scripts/create_admin_user.py
```

---

## 🔍 **Health Checks & Monitoring**

### **Service Health Endpoints**
```bash
# Health check endpoints
curl http://localhost:8000/health
curl http://localhost:8000/metrics
curl http://localhost:9090/-/healthy  # Prometheus
```

### **Grafana Dashboard Access**
- **URL:** http://localhost:3000
- **Username:** admin  
- **Password:** (from GRAFANA_PASSWORD env var)

**Key Dashboards:**
- Heart Protocol System Overview
- Healing Metrics Dashboard
- Crisis Intervention Monitoring
- Community Health Indicators
- Privacy Protection Metrics

### **Log Management**
```bash
# View logs
docker-compose logs -f heart-protocol-api
docker-compose logs -f monarch-bot
docker-compose logs -f care-monitor

# Log locations
./logs/heart-protocol.log
./logs/monarch-bot.log
./logs/care-monitor.log
```

---

## 🌐 **Bluesky Integration Setup**

### **1. Create Bluesky Account**
1. Create account: `monarch.bsky.social` (or your chosen handle)
2. Generate App Password in Bluesky settings
3. Add to `.env` file as `BLUESKY_APP_PASSWORD`

### **2. Register Feed Generators**
```bash
# Register feeds with Bluesky
docker-compose exec heart-protocol-api python scripts/register_feeds.py

# Verify feed registration
curl "https://bsky.app/profile/monarch.bsky.social/feed/daily-gentle-reminders"
```

### **3. Bot Account Setup**
```bash
# Start Monarch Bot
docker-compose exec monarch-bot python -m heart_protocol.bot.monarch_bot

# Test bot functionality
docker-compose exec monarch-bot python scripts/test_bot_responses.py
```

---

## 🔒 **Security Configuration**

### **Firewall Setup**
```bash
# UFW firewall configuration
sudo ufw allow 22/tcp      # SSH
sudo ufw allow 80/tcp      # HTTP
sudo ufw allow 443/tcp     # HTTPS
sudo ufw enable

# Close development ports in production
# sudo ufw deny 5432/tcp   # PostgreSQL (only allow local)
# sudo ufw deny 6379/tcp   # Redis (only allow local)
```

### **Rate Limiting**
```nginx
# Nginx rate limiting (already configured)
limit_req_zone $binary_remote_addr zone=api:10m rate=60r/m;
limit_req_zone $binary_remote_addr zone=crisis:10m rate=10r/s;
```

### **Privacy Protection**
```python
# Verify privacy settings
PRIVACY_MODE=strict
ANONYMIZATION_ENABLED=true
CONSENT_REQUIRED=true
DATA_RETENTION_DAYS=90
```

---

## 📊 **Scaling Configuration**

### **Horizontal Scaling**
```yaml
# docker-compose.scale.yml
version: '3.8'
services:
  heart-protocol-api:
    deploy:
      replicas: 3
  
  care-monitor:
    deploy:
      replicas: 2
```

### **Database Scaling**
```bash
# PostgreSQL optimization
docker-compose exec postgres psql -U heart_user -d heart_protocol
```

```sql
-- Optimize for production
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET work_mem = '4MB';
SELECT pg_reload_conf();
```

### **Redis Optimization**
```bash
# Redis memory optimization
echo 'maxmemory 512mb' >> redis.conf
echo 'maxmemory-policy allkeys-lru' >> redis.conf
```

---

## 🧪 **Testing Deployment**

### **Smoke Tests**
```bash
# Run deployment smoke tests
docker-compose exec heart-protocol-api python -m pytest tests/deployment/

# Test feed generation
curl http://localhost:8000/feeds/daily-gentle-reminders

# Test care detection
curl -X POST http://localhost:8000/api/care-detection \
  -H "Content-Type: application/json" \
  -d '{"text": "I am feeling overwhelmed today"}'

# Test crisis intervention
curl -X POST http://localhost:8000/api/crisis-assessment \
  -H "Content-Type: application/json" \
  -d '{"text": "I don't want to be here anymore"}'
```

### **Integration Tests**
```bash
# Full integration test suite
docker-compose exec heart-protocol-api python -m pytest tests/integration/ -v

# Cultural sensitivity tests
docker-compose exec heart-protocol-api python -m pytest tests/cultural_sensitivity_tests.py

# Privacy protection tests
docker-compose exec heart-protocol-api python -m pytest tests/privacy_protection_tests.py
```

---

## 🔧 **Maintenance**

### **Regular Updates**
```bash
# Update container images
docker-compose pull
docker-compose up -d

# Database backups
docker-compose exec postgres pg_dump -U heart_user heart_protocol > backup_$(date +%Y%m%d).sql

# Log rotation
docker-compose exec heart-protocol-api logrotate /etc/logrotate.conf
```

### **Monitoring Alerts**
```yaml
# alerts.yml for Prometheus
groups:
  - name: heart-protocol
    rules:
      - alert: HighCrisisVolume
        expr: rate(crisis_interventions_total[5m]) > 10
        for: 2m
        annotations:
          summary: "High crisis intervention volume detected"
          
      - alert: LowHealingEffectiveness
        expr: avg(healing_effectiveness_score) < 75
        for: 10m
        annotations:
          summary: "Healing effectiveness below threshold"
```

---

## 🌍 **Multi-Region Deployment**

### **Global Load Balancing**
```bash
# Deploy to multiple regions
docker-compose -f docker-compose.yml -f docker-compose.us-east.yml up -d
docker-compose -f docker-compose.yml -f docker-compose.eu-west.yml up -d
```

### **Cultural Localization**
```bash
# Add language support
docker-compose exec heart-protocol-api python scripts/add_language_support.py --language es
docker-compose exec heart-protocol-api python scripts/add_language_support.py --language fr
```

---

## 🚨 **Troubleshooting**

### **Common Issues**

**1. Bluesky Connection Issues:**
```bash
# Check Bluesky credentials
docker-compose logs monarch-bot | grep -i "authentication"

# Test AT Protocol connection
curl -X POST https://bsky.social/xrpc/com.atproto.server.createSession \
  -H "Content-Type: application/json" \
  -d '{"identifier": "monarch.bsky.social", "password": "your-app-password"}'
```

**2. Database Connection Issues:**
```bash
# Check database connectivity
docker-compose exec heart-protocol-api python -c "from heart_protocol.core import db; print(db.test_connection())"

# Reset database
docker-compose down
docker volume rm heart-protocol_postgres_data
docker-compose up -d postgres
```

**3. Memory Issues:**
```bash
# Check memory usage
docker stats

# Increase memory limits
docker-compose -f docker-compose.yml -f docker-compose.override.yml up -d
```

### **Debug Mode**
```bash
# Enable debug logging
export DEBUG=true
export LOG_LEVEL=DEBUG
docker-compose up -d

# View debug logs
docker-compose logs -f heart-protocol-api | grep DEBUG
```

---

## 📱 **Mobile App Deployment** (Future)

### **React Native Setup** (Planned)
```bash
# Mobile app deployment (coming soon)
git clone https://github.com/your-org/heart-protocol-mobile.git
cd heart-protocol-mobile
npm install
npm run deploy:ios
npm run deploy:android
```

---

## 🤝 **Community Deployment Support**

### **Getting Help**
- **Documentation:** [docs.heart-protocol.org](https://docs.heart-protocol.org)
- **Community Discord:** [discord.gg/heart-protocol](https://discord.gg/heart-protocol)
- **GitHub Issues:** [github.com/your-org/heart-protocol/issues](https://github.com/your-org/heart-protocol/issues)
- **Emergency Support:** deployment-help@heart-protocol.org

### **Deployment Service**
For organizations needing deployment assistance:
- **Community Deployments:** Free for nonprofits and educational institutions
- **Enterprise Deployments:** White-glove deployment service available
- **Managed Hosting:** Fully managed Heart Protocol hosting available

---

**💙 Deploy with care. Every deployment brings healing algorithms to more communities.**

*"Technology that serves love, deployed with the same care we give to every interaction."*