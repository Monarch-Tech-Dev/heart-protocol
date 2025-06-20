# 🚨 Heart Protocol Security Incident Response

## **Immediate Response (First 15 minutes)**

### **If Secrets Compromised:**
```bash
# 1. IMMEDIATELY rotate all secrets
./scripts/rotate_secrets.sh

# 2. Force restart all services
docker-compose down
docker-compose up -d

# 3. Check for unauthorized access
grep "Failed password" /var/log/auth.log
docker-compose logs | grep -i "unauthorized\|error\|failed"
```

### **If Server Compromised:**
```bash
# 1. Disconnect from internet
sudo ufw deny out

# 2. Create forensic backup
sudo dd if=/dev/vda of=/backup/forensic-$(date +%Y%m%d).img

# 3. Check for intrusions
sudo aide --check
sudo rkhunter --check
```

### **If Application Breach:**
```bash
# 1. Enable maintenance mode
echo "MAINTENANCE_MODE=true" >> .env
docker-compose restart

# 2. Secure the database
docker-compose exec postgres psql -U heart_user -c "\q"
docker-compose stop postgres

# 3. Review access logs
tail -f logs/heart-protocol.log | grep -E "POST|PUT|DELETE"
```

## **Investigation Phase (Next 1 hour)**

### **Forensic Analysis:**
1. Preserve log files before rotation
2. Identify attack vector
3. Assess data exposure scope
4. Document timeline of events
5. Collect evidence for potential law enforcement

### **Damage Assessment:**
- User data potentially accessed
- Systems compromised
- Services affected
- Data integrity status

## **Recovery Phase (Next 24 hours)**

### **System Recovery:**
1. Patch vulnerabilities
2. Restore from clean backups
3. Implement additional security measures
4. Full security audit
5. Gradual service restoration

### **Communication:**
- Notify affected users within 72 hours
- Coordinate with authorities if required
- Public transparency report
- Update security documentation

## **Post-Incident (Following weeks)**

### **Improvements:**
1. Security architecture review
2. Enhanced monitoring implementation
3. Team security training
4. Updated incident response procedures
5. Regular security audits

## **Emergency Contacts**

- **Security Team:** security@heart-protocol.org
- **Infrastructure:** infrastructure@heart-protocol.org  
- **Legal:** legal@heart-protocol.org
- **Community:** community@heart-protocol.org

## **24/7 Crisis Escalation**

- **Severity 1 (Critical):** Page on-call engineer immediately
- **Severity 2 (High):** Notify within 1 hour
- **Severity 3 (Medium):** Notify within 4 hours
- **Severity 4 (Low):** Include in daily security report