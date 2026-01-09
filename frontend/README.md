# Techmania Dashboard

Moderní dashboard pro analýzu a predikci návštěvnosti science centra Techmania.

## Technologie

- **Next.js 14** - React framework s App Router
- **TypeScript** - Typová bezpečnost
- **Tailwind CSS** - Utility-first CSS framework
- **Headless UI** - Komponenty bez stylování
- **Chart.js** - Vizualizace dat
- **date-fns** - Práce s datumy

## Instalace

```bash
# Instalace závislostí
npm install

# Spuštění vývojového serveru
npm run dev
```

Aplikace běží na [http://localhost:3000](http://localhost:3000)

## Konfigurace

Vytvořte soubor `.env.local` s následující konfigurací:

```env
NEXT_PUBLIC_API_URL=http://localhost:5000
```

## Funkce

### Dashboard
- Přehled klíčových metrik
- Vizualizace návštěvnosti
- Srovnání skutečné vs. predikované návštěvnosti

### Predikce
- **Jednoduchá predikce** - Pro konkrétní den s parametry
- **Rozsahová predikce** - Pro více dní najednou

### Parametry predikce
- Datum
- Státní svátek / prázdniny
- Otevírací doba

## API Endpointy

### POST /predict
Predikce pro jeden den

```json
{
  "date": "2026-01-15",
  "is_holiday": false,
  "opening_hours": "09:00-17:00"
}
```

### POST /predict/range
Predikce pro rozsah dat

```json
{
  "start_date": "2026-01-15",
  "end_date": "2026-01-20"
}
```

### GET /health
Health check API

## Struktura projektu

```
frontend/
├── src/
│   ├── app/              # Next.js App Router
│   │   ├── layout.tsx    # Root layout
│   │   ├── page.tsx      # Hlavní stránka
│   │   └── globals.css   # Globální styly
│   ├── components/       # React komponenty
│   │   ├── Sidebar.tsx
│   │   ├── StatsCards.tsx
│   │   ├── VisitorChart.tsx
│   │   ├── PredictionForm.tsx
│   │   └── RangePredictionForm.tsx
│   ├── lib/              # Utility funkce
│   │   └── api.ts        # API klient
│   └── types/            # TypeScript typy
│       └── api.ts
├── public/               # Statické soubory
├── package.json
├── tsconfig.json
├── tailwind.config.ts
└── next.config.js
```

## Deployment

```bash
# Build produkční verze
npm run build

# Spuštění produkční verze
npm start
```

## Licence

MIT
