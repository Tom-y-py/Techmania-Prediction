# Frontend Dockerfile pro Techmania Prediction

# Stage 1: Build
FROM node:18-alpine AS builder

WORKDIR /app

# Kopírování package files
COPY package*.json ./

# Instalace závislostí
RUN npm install

# Kopírování zdrojových souborů a .env souboru
COPY . .
COPY .env.production .env.production

# Build-time argumenty pro Next.js
ARG NEXT_PUBLIC_API_URL=https://techmania.korex.space/api
ARG NEXT_PUBLIC_ENVIRONMENT=production
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL
ENV NEXT_PUBLIC_ENVIRONMENT=$NEXT_PUBLIC_ENVIRONMENT

# Build aplikace pro produkci
RUN npm run build

# Stage 2: Production
FROM node:18-alpine AS runner

WORKDIR /app

# Nastavení prostředí na production
ENV NODE_ENV=production
ENV NEXT_PUBLIC_API_URL=https://techmania.korex.space/api
ENV NEXT_PUBLIC_ENVIRONMENT=production

# Vytvoření non-root uživatele
RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

# Kopírování build artifacts z builder stage
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static

# Nastavení vlastníka souborů
RUN chown -R nextjs:nodejs /app

# Přepnutí na non-root uživatele
USER nextjs

# Exponování portu
EXPOSE 3000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD node -e "require('http').get('http://localhost:3000', (r) => {process.exit(r.statusCode === 200 ? 0 : 1)})"

# Spuštění aplikace
CMD ["node", "server.js"]
