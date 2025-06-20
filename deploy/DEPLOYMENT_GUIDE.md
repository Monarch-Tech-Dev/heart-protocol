# 🚀 Heart Protocol Deployment Guide

## Quick Start (Recommended)

### Option 1: $5 VPS - 24/7 Operation

**Best for:** Getting started quickly, low cost

1. **Get a VPS:**
   - DigitalOcean Droplet ($5/month) 
   - Linode Nanode ($5/month)
   - Hetzner CX11 ($4/month)

2. **Deploy in 5 minutes:**
   ```bash
   # SSH to your VPS
   ssh root@your-vps-ip
   
   # Run deployment script
   curl -sSL https://raw.githubusercontent.com/your-repo/Heart-Protocol/main/deploy/vps_setup.sh | bash
   
   # Copy your .env file
   scp .env root@your-vps-ip:/root/Heart-Protocol/
   
   # Start the service
   sudo systemctl start monarch-bot
   ```

3. **Monitor:**
   ```bash
   # Check status
   sudo systemctl status monarch-bot
   
   # View logs
   sudo journalctl -u monarch-bot -f
   ```

### Option 2: Docker Deployment

**Best for:** Easy scaling and portability

1. **Local testing:**
   ```bash
   # Build and run
   docker-compose up --build
   
   # With monitoring
   docker-compose --profile monitoring up
   ```

2. **Deploy to cloud:**
   - **AWS ECS/Fargate**
   - **Google Cloud Run** 
   - **DigitalOcean App Platform**
   - **Railway.app**
   - **Render.com**

### Option 3: Cloud Platforms

#### Railway.app (Easiest)
1. Connect GitHub repo
2. Add environment variables
3. Deploy automatically

#### Render.com
1. Connect GitHub repo  
2. Choose "Web Service"
3. Add environment variables
4. Deploy

#### DigitalOcean App Platform
1. Connect GitHub repo
2. Configure build settings
3. Add environment variables  
4. Deploy

#### AWS ECS (Most scalable)
1. Build Docker image
2. Push to ECR
3. Create ECS task definition
4. Deploy to Fargate

## Environment Variables Needed

```bash
BLUESKY_HANDLE=monarchbot.bsky.social
BLUESKY_APP_PASSWORD=your-app-password
ENVIRONMENT=production
LOG_LEVEL=INFO
```

## Monitoring Setup

### Basic Monitoring (Free)
- **UptimeRobot** - Monitor if bot is online
- **Systemd logs** - Built-in logging on VPS

### Advanced Monitoring (Optional)
- **Prometheus + Grafana** - Metrics and dashboards
- **Sentry** - Error tracking
- **Datadog** - Full observability

## Cost Breakdown

| Option | Monthly Cost | Pros | Cons |
|--------|-------------|------|------|
| VPS | $5 | Simple, full control | Manual setup |
| Railway | $5-10 | Easy deployment | Less control |
| Render | $7 | Good free tier | Limited customization |
| AWS | $10-50 | Highly scalable | Complex setup |

## Security Checklist

- [ ] Use environment variables for secrets
- [ ] Enable firewall on VPS
- [ ] Use non-root user for app
- [ ] Enable automatic security updates
- [ ] Monitor for failed login attempts
- [ ] Regular backups of logs/data

## Scaling Considerations

### Current (Single Bot):
- 1 CPU, 1GB RAM sufficient
- Handles ~1000 users comfortably

### Future Scaling:
- **Multiple regions** for global users
- **Load balancing** for high availability  
- **Database** for user preferences
- **Redis** for caching and rate limiting
- **CDN** for static assets

## Emergency Procedures

### Bot Goes Down:
1. Check systemd status: `sudo systemctl status monarch-bot`
2. Check logs: `sudo journalctl -u monarch-bot -f`
3. Restart: `sudo systemctl restart monarch-bot`
4. Check Bluesky API status

### High CPU/Memory:
1. Monitor with `htop`
2. Check for infinite loops in logs
3. Restart service if needed
4. Scale up resources if necessary

### Crisis Situation:
1. Bot continues operating for crisis intervention
2. Human escalation protocols activate
3. Log all crisis interactions for review
4. Never shut down during active crisis

## Backup Strategy

### Code:
- GitHub repository (primary)
- Automated daily backups

### Logs:
- Rotate logs daily
- Keep 30 days locally
- Archive to cloud storage monthly

### Configuration:
- Environment variables documented
- Deployment scripts version controlled
- Recovery procedures tested monthly

## Next Steps After Deployment

1. **Monitor first 24 hours** closely
2. **Test crisis intervention** flow
3. **Engage with community** gently
4. **Collect feedback** for improvements
5. **Scale based on usage** patterns