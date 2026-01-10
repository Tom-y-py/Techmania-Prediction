#!/bin/bash

# Deployment skript pro Techmania Prediction
# Autor: Setup automation
# Datum: 2026-01-10

set -e

echo "🚀 Spouštím deployment Techmania Prediction..."

# Kontrola, že jsme ve správném adresáři
if [ ! -f "docker-compose.yml" ]; then
    echo "❌ Chyba: docker-compose.yml nebyl nalezen!"
    echo "Ujistěte se, že jste ve správném adresáři."
    exit 1
fi

# Kontrola SSL certifikátů
if [ ! -f "/etc/letsencrypt/live/techmania.korex.space/fullchain.pem" ]; then
    echo "⚠️  SSL certifikáty nebyly nalezeny!"
    echo "Použiji HTTP-only konfiguraci nginx."
    
    # Použití HTTP-only konfigurace
    if [ -f "nginx/nginx-http-only.conf" ]; then
        cp nginx/nginx-http-only.conf nginx/nginx.conf
    fi
else
    echo "✅ SSL certifikáty nalezeny"
fi

# Zastavení běžících containerů
echo "🛑 Zastavuji běžící containery..."
docker-compose down

# Build a spuštění
echo "🔨 Buildím Docker images..."
docker-compose build --no-cache

echo "▶️  Spouštím containery..."
docker-compose up -d

# Čekání na zdravý stav
echo "⏳ Čekám na spuštění služeb..."
sleep 10

# Kontrola zdraví služeb
echo "🏥 Kontroluji zdraví služeb..."

# Backend health check
BACKEND_HEALTH=$(docker-compose exec -T backend curl -s -o /dev/null -w "%{http_code}" http://localhost:5000/health || echo "000")
if [ "$BACKEND_HEALTH" = "200" ]; then
    echo "✅ Backend je zdravý (HTTP $BACKEND_HEALTH)"
else
    echo "⚠️  Backend neodpovídá správně (HTTP $BACKEND_HEALTH)"
fi

# Frontend health check
FRONTEND_HEALTH=$(docker-compose exec -T frontend curl -s -o /dev/null -w "%{http_code}" http://localhost:3000 || echo "000")
if [ "$FRONTEND_HEALTH" = "200" ]; then
    echo "✅ Frontend je zdravý (HTTP $FRONTEND_HEALTH)"
else
    echo "⚠️  Frontend neodpovídá správně (HTTP $FRONTEND_HEALTH)"
fi

echo ""
echo "✅ Deployment dokončen!"
echo ""
echo "📊 Status služeb:"
docker-compose ps
echo ""
echo "🌐 Aplikace běží na:"
if [ -f "/etc/letsencrypt/live/techmania.korex.space/fullchain.pem" ]; then
    echo "   https://techmania.korex.space/"
else
    echo "   http://techmania.korex.space/"
fi
echo ""
echo "📝 Logy můžete sledovat příkazem:"
echo "   docker-compose logs -f"
