# Techmania Dashboard - Kompletní dokumentace

## 📋 Přehled projektu

Moderní Next.js dashboard pro science centrum Techmania v Plzni. Aplikace umožňuje analýzu historických dat a predikci budoucí návštěvnosti pomocí pokročilých strojového učení modelů.

## 🚀 Rychlý start

### 1. Instalace
```bash
cd frontend
npm install
```

### 2. Konfigurace
Vytvořte `.env.local`:
```env
NEXT_PUBLIC_API_URL=http://localhost:5000
```

### 3. Spuštění
```bash
npm run dev
```

Dashboard běží na: **http://localhost:3000**

## 🏗️ Architektura

### Frontend Stack
- **Next.js 14** - React framework s App Router
- **TypeScript** - Typová bezpečnost
- **Tailwind CSS** - Utility-first styling
- **Headless UI** - Přístupné komponenty
- **Chart.js** - Data vizualizace
- **date-fns** - Práce s datumy

### Struktura projektu
```
frontend/
├── src/
│   ├── app/                    # Next.js App Router
│   │   ├── layout.tsx          # Root layout
│   │   ├── page.tsx            # Dashboard hlavní stránka
│   │   ├── loading.tsx         # Loading state
│   │   ├── error.tsx           # Error boundary
│   │   ├── not-found.tsx       # 404 stránka
│   │   └── globals.css         # Globální styly
│   │
│   ├── components/             # React komponenty
│   │   ├── Sidebar.tsx         # Navigace
│   │   ├── Header.tsx          # Hlavička s uživatelem
│   │   ├── HealthStatus.tsx    # API status indikátor
│   │   ├── StatsCards.tsx      # Statistické karty
│   │   ├── VisitorChart.tsx    # Graf návštěvnosti
│   │   ├── PredictionForm.tsx  # Formulář jednoduchá predikce
│   │   ├── RangePredictionForm.tsx  # Formulář rozsahová predikce
│   │   ├── ExportButton.tsx    # Export dat (CSV, JSON)
│   │   ├── Notification.tsx    # Notifikační dialog
│   │   ├── LoadingSpinner.tsx  # Loading komponenta
│   │   └── index.ts            # Exporty
│   │
│   ├── lib/                    # Utility funkce
│   │   ├── api.ts              # API klient
│   │   └── utils.ts            # Pomocné funkce
│   │
│   └── types/                  # TypeScript definice
│       └── api.ts              # API typy
│
├── public/                     # Statické soubory
│   ├── favicon.svg
│   └── manifest.json
│
├── package.json                # Závislosti
├── tsconfig.json              # TypeScript konfigurace
├── tailwind.config.ts         # Tailwind konfigurace
├── next.config.js             # Next.js konfigurace
├── postcss.config.js          # PostCSS konfigurace
├── .env.local                 # Lokální proměnné prostředí
├── .gitignore
├── README.md
└── SETUP.md
```

## 🎨 Komponenty

### Layout komponenty

#### `Sidebar.tsx`
- Responzivní navigační menu
- Mobile: Dialog overlay
- Desktop: Fixní sidebar
- Navigace: Dashboard, Predikce, Analýza, Nastavení

#### `Header.tsx`
- Hlavička aplikace
- Health status indikátor
- Notifikace
- Uživatelské menu

#### `HealthStatus.tsx`
- Real-time monitoring API
- Auto-refresh každých 30s
- Vizuální indikace stavu (zelená/červená)

### Data komponenty

#### `StatsCards.tsx`
- 4 statistické karty:
  - Celkový počet návštěvníků
  - Průměr návštěvníků/den
  - Den s nejvyšší návštěvností
  - Měsíční trend
- Barevné ikony (Heroicons)
- Hover efekty

#### `VisitorChart.tsx`
- Line chart s Chart.js
- Srovnání skutečné vs. predikované návštěvnosti
- Interaktivní tooltips
- Responzivní design
- 400px výška

### Predikce komponenty

#### `PredictionForm.tsx`
**Vstupy:**
- Datum (date picker)
- Státní svátek (checkbox)
- Otevírací doba (select)

**Výstup:**
- Predikovaný počet návštěvníků
- Confidence interval (dolní a horní mez)
- Formátované datum (česky)

**Funkce:**
- Form validace
- Loading states
- Error handling
- Gradient pozadí výsledku

#### `RangePredictionForm.tsx`
**Vstupy:**
- Datum od
- Datum do

**Výstup:**
- Tabulka predikovaných hodnot
- Celková predikce pro období
- Export tlačítko (CSV, JSON)

**Funkce:**
- Grid layout
- Formátování dat
- Export funkcionalita

### Utility komponenty

#### `ExportButton.tsx`
- Menu s dropdown
- Export CSV (UTF-8 s BOM)
- Export JSON
- Loading states
- Auto-download

#### `Notification.tsx`
- Dialog overlay (Headless UI)
- 3 typy: success, error, info
- Barevné ikony
- Animované přechody

#### `LoadingSpinner.tsx`
- 3 velikosti: sm, md, lg
- Spinning animace
- Techmania blue barva

## 🎨 Design System

### Barvy
```typescript
colors: {
  techmania: {
    blue: '#0066CC',    // Primární barva
    green: '#00CC66',   // Akcentová barva
    dark: '#1a1a2e',    // Tmavá
    gray: '#16213e',    // Šedá
  }
}
```

### Typografie
- Font: Inter (Google Fonts)
- Weights: 300-900

### Spacing
- Tailwind default scale
- Gap systém: 4, 6, 8

### Shadows
- sm: Cards hover
- lg: Modals

## 🔌 API Integration

### Endpoints

#### `POST /predict`
Jednoduchá predikce pro jeden den

**Request:**
```json
{
  "date": "2026-01-15",
  "is_holiday": false,
  "opening_hours": "09:00-17:00"
}
```

**Response:**
```json
{
  "date": "2026-01-15",
  "predicted_visitors": 3542,
  "confidence_interval": {
    "lower": 3100,
    "upper": 4000
  }
}
```

#### `POST /predict/range`
Rozsahová predikce

**Request:**
```json
{
  "start_date": "2026-01-15",
  "end_date": "2026-01-20"
}
```

**Response:**
```json
{
  "predictions": [
    {
      "date": "2026-01-15",
      "predicted_visitors": 3542
    },
    ...
  ],
  "total_predicted": 21250
}
```

#### `GET /health`
Health check

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### API Client (`src/lib/api.ts`)

```typescript
export const api = {
  async predict(data: PredictionRequest): Promise<PredictionResponse>
  async predictRange(data: RangePredictionRequest): Promise<RangePredictionResponse>
  async healthCheck(): Promise<{ status: string; model_loaded: boolean }>
}
```

**Features:**
- Error handling
- Type safety
- Environment-based URL
- JSON content-type headers

## 📱 Responzivní design

### Breakpoints
- Mobile: < 640px
- Tablet: 640px - 1024px
- Desktop: > 1024px

### Adaptace
- **Sidebar**: Mobile dialog → Desktop fixed
- **Grid**: 1 col → 2 cols → 4 cols
- **Forms**: Stack → Grid
- **Charts**: Full width, auto-height

## 🚀 Production Build

### Build
```bash
npm run build
```

### Start production
```bash
npm start
```

### Optimalizace
- Automatic code splitting
- Image optimization
- Static generation kde možné
- CSS purging (Tailwind)

## 🔧 Konfigurace

### Environment variables
```env
NEXT_PUBLIC_API_URL=http://localhost:5000
```

### Next.js config
```javascript
{
  reactStrictMode: true
}
```

### Tailwind custom theme
- Custom colors (techmania)
- Extended color palette
- Custom utilities

## 🧪 Testing

### Manuální testování
1. Spusťte backend API
2. Spusťte frontend (`npm run dev`)
3. Otevřete http://localhost:3000
4. Testujte jednotlivé funkce

### Checklist
- [ ] Health status zobrazuje "API připojeno"
- [ ] Stats cards zobrazují data
- [ ] Graf se vykreslí
- [ ] Jednoduchá predikce funguje
- [ ] Rozsahová predikce funguje
- [ ] Export CSV/JSON funguje
- [ ] Responzivní na mobile
- [ ] Error handling funguje

## 📦 Deployment

### Vercel (doporučeno)
```bash
npm install -g vercel
vercel
```

### Docker
```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

### Environment
Nastavte `NEXT_PUBLIC_API_URL` na produkční API URL

## 🐛 Troubleshooting

### API není dostupné
1. Zkontrolujte backend běží
2. Ověřte `.env.local`
3. Kontrolujte CORS nastavení

### Build chyby
```bash
rm -rf .next node_modules
npm install
npm run build
```

### Port conflicts
```bash
PORT=3001 npm run dev
```

## 📄 Licence

MIT

## 👥 Autoři

Vytvořeno pro Techmanii - Science centrum Plzeň

---

**Verze:** 1.0.0  
**Poslední aktualizace:** 9. ledna 2026
