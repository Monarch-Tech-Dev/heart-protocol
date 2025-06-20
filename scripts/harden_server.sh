#!/bin/bash
# Heart Protocol Server Security Hardening Script
# Run this on your production server to enhance security

set -e

echo "🛡️  Heart Protocol Server Hardening"
echo "===================================="

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "❌ Please run as root (sudo ./harden_server.sh)"
    exit 1
fi

echo "🔒 Setting up firewall..."

# Install and configure UFW firewall
apt update
apt install -y ufw fail2ban

# Default firewall rules
ufw default deny incoming
ufw default allow outgoing

# Allow essential services
ufw allow 22/tcp comment 'SSH'
ufw allow 80/tcp comment 'HTTP' 
ufw allow 443/tcp comment 'HTTPS'

# Block internal services from external access
ufw deny 5432/tcp comment 'PostgreSQL (internal only)'
ufw deny 6379/tcp comment 'Redis (internal only)'
ufw deny 9090/tcp comment 'Prometheus (internal only)'

# Enable firewall
ufw --force enable

echo "✅ Firewall configured"

echo "🔐 Configuring fail2ban..."

# Configure fail2ban for SSH protection
cat > /etc/fail2ban/jail.local << 'EOF'
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 3

[sshd]
enabled = true
port = ssh
filter = sshd
logpath = /var/log/auth.log
maxretry = 3
bantime = 3600

[heart-protocol-api]
enabled = true
port = 8000
filter = heart-protocol
logpath = /opt/heart-protocol/logs/heart-protocol.log
maxretry = 10
bantime = 1800
EOF

# Create custom fail2ban filter for Heart Protocol
mkdir -p /etc/fail2ban/filter.d
cat > /etc/fail2ban/filter.d/heart-protocol.conf << 'EOF'
[Definition]
failregex = ^.*\[.*\] ".*" 429 .* - .*$
            ^.*Rate limit exceeded.*IP: <HOST>.*$
            ^.*Suspicious activity detected.*IP: <HOST>.*$
ignoreregex =
EOF

# Start fail2ban
systemctl enable fail2ban
systemctl restart fail2ban

echo "✅ Fail2ban configured"

echo "🗃️  Setting up secure directories..."

# Create Heart Protocol directory with proper permissions
mkdir -p /opt/heart-protocol
mkdir -p /opt/heart-protocol/logs
mkdir -p /opt/heart-protocol/data
mkdir -p /opt/heart-protocol/backups

# Set secure permissions
chown -R root:docker /opt/heart-protocol
chmod 750 /opt/heart-protocol
chmod 700 /opt/heart-protocol/data
chmod 700 /opt/heart-protocol/backups

echo "✅ Directories secured"

echo "🔄 Configuring automatic security updates..."

# Install automatic security updates
apt install -y unattended-upgrades

# Configure automatic security updates
cat > /etc/apt/apt.conf.d/50unattended-upgrades << 'EOF'
Unattended-Upgrade::Allowed-Origins {
    "${distro_id}:${distro_codename}-security";
    "${distro_id}ESMApps:${distro_codename}-apps-security";
    "${distro_id}ESM:${distro_codename}-infra-security";
};

Unattended-Upgrade::AutoFixInterruptedDpkg "true";
Unattended-Upgrade::MinimalSteps "true";
Unattended-Upgrade::Remove-Unused-Dependencies "true";
Unattended-Upgrade::Automatic-Reboot "false";
EOF

# Enable automatic updates
echo 'APT::Periodic::Update-Package-Lists "1";
APT::Periodic::Download-Upgradeable-Packages "1";
APT::Periodic::AutocleanInterval "7";
APT::Periodic::Unattended-Upgrade "1";' > /etc/apt/apt.conf.d/20auto-upgrades

echo "✅ Automatic security updates enabled"

echo "📊 Setting up log rotation..."

# Configure log rotation for Heart Protocol
cat > /etc/logrotate.d/heart-protocol << 'EOF'
/opt/heart-protocol/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 root root
    postrotate
        /usr/bin/docker-compose -f /opt/heart-protocol/docker-compose.yml restart heart-protocol-api > /dev/null 2>&1 || true
    endscript
}
EOF

echo "✅ Log rotation configured"

echo "🔐 Hardening SSH..."

# Backup original SSH config
cp /etc/ssh/sshd_config /etc/ssh/sshd_config.backup

# Harden SSH configuration
sed -i 's/#PermitRootLogin yes/PermitRootLogin no/' /etc/ssh/sshd_config
sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
sed -i 's/#PubkeyAuthentication yes/PubkeyAuthentication yes/' /etc/ssh/sshd_config
sed -i 's/X11Forwarding yes/X11Forwarding no/' /etc/ssh/sshd_config

# Add additional security settings
echo "
# Heart Protocol Security Hardening
Protocol 2
MaxAuthTries 3
ClientAliveInterval 300
ClientAliveCountMax 2
UseDNS no
AllowUsers $(logname)
" >> /etc/ssh/sshd_config

# Restart SSH
systemctl restart sshd

echo "✅ SSH hardened"

echo "🔍 Setting up intrusion detection..."

# Install and configure AIDE
apt install -y aide
aideinit
mv /var/lib/aide/aide.db.new /var/lib/aide/aide.db

# Create daily AIDE check
cat > /etc/cron.daily/aide-check << 'EOF'
#!/bin/bash
aide --check > /var/log/aide.log 2>&1
if [ $? -ne 0 ]; then
    echo "AIDE detected changes on $(hostname)" | mail -s "AIDE Alert" admin@your-domain.com
fi
EOF
chmod +x /etc/cron.daily/aide-check

echo "✅ Intrusion detection setup"

echo ""
echo "🎉 Server hardening complete!"
echo ""
echo "📋 SECURITY CHECKLIST:"
echo "   ✅ Firewall configured (UFW)"
echo "   ✅ Fail2ban installed and configured"
echo "   ✅ Automatic security updates enabled"
echo "   ✅ SSH hardened"
echo "   ✅ Log rotation configured"
echo "   ✅ Secure directories created"
echo "   ✅ Intrusion detection (AIDE) setup"
echo ""
echo "🔐 NEXT STEPS:"
echo "   1. Add your SSH public key to ~/.ssh/authorized_keys"
echo "   2. Test SSH access with key-based auth"
echo "   3. Deploy Heart Protocol: cd /opt/heart-protocol && ./deploy.sh"
echo "   4. Set up SSL certificates: certbot --nginx"
echo "   5. Configure monitoring alerts"
echo ""
echo "⚠️  IMPORTANT:"
echo "   - Test SSH access before logging out!"
echo "   - Keep your private keys secure"
echo "   - Monitor /var/log/auth.log for suspicious activity"
echo "   - Run 'ufw status' to verify firewall rules"
echo ""
echo "💚 Your Heart Protocol server is now hardened! 🛡️"