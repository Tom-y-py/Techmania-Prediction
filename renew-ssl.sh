#!/bin/bash

# Skript pro obnovení SSL certifikátů pro Techmania Prediction
# Tento skript lze přidat do cronu pro automatické obnovení

set -e

echo "🔄 Obnova SSL certifikátů pro techmania.korex.space..."

# Zastavení nginx pro obnovení (certbot potřebuje port 80)
echo "🛑 Zastavuji nginx..."
docker-compose stop nginx

# Obnovení certifikátu
echo "📜 Obnovuji certifikát..."
sudo certbot renew --standalone --preferred-challenges http

# Restart nginx s novými certifikáty
echo "▶️  Restartuji nginx..."
docker-compose start nginx

echo "✅ Certifikáty byly obnoveny!"
echo ""
echo "📅 Platnost certifikátu:"
sudo certbot certificates -d techmania.korex.space
