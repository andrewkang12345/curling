#!/usr/bin/env bash
# One-command HTTPS for the arena once you own a domain.
#   1. Point an A record for YOUR.DOMAIN at this instance's public IP.
#   2. sudo bash scripts/setup_tls.sh YOUR.DOMAIN
# Installs certbot, rewrites the nginx server block for the domain, obtains a
# Let's Encrypt cert (auto-renewing via certbot.timer), redirects HTTP->HTTPS.
set -euo pipefail
DOMAIN="${1:?usage: setup_tls.sh your.domain.com}"
dnf install -y certbot python3-certbot-nginx >/dev/null
CONF=/etc/nginx/conf.d/curling-game.conf
sed -i "s/server_name .*/server_name ${DOMAIN};/" "$CONF" || true
grep -q "server_name ${DOMAIN}" "$CONF" || sed -i "s/listen 80;/listen 80;\n    server_name ${DOMAIN};/" "$CONF"
nginx -t && systemctl reload nginx
certbot --nginx -d "$DOMAIN" --redirect --non-interactive --agree-tos -m admin@"${DOMAIN}"
systemctl enable --now certbot-renew.timer 2>/dev/null || true
echo "DONE: https://${DOMAIN} (cert auto-renews)"
