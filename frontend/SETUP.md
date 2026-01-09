# Návod na spuštění Techmania Dashboard

## Předpoklady

- Node.js 18+ nainstalovaný
- npm nebo yarn package manager

## Rychlé spuštění

### 1. Instalace závislostí

```bash
cd frontend
npm install
```

### 2. Konfigurace prostředí

Vytvořte soubor `.env.local` v kořenové složce `frontend/`:

```env
NEXT_PUBLIC_API_URL=http://localhost:5000
```

### 3. Spuštění vývojového serveru

```bash
npm run dev
```

Aplikace poběží na: http://localhost:3000

### 4. Spuštění backend API (v samostatném terminálu)

```bash
cd ..
python app/app.py
```

Backend API poběží na: http://localhost:5000

## Dostupné příkazy

```bash
# Vývojový server
npm run dev

# Build produkční verze
npm run build

# Spuštění produkční verze
npm start

# Kontrola kódu
npm run lint
```

## Funkce dashboardu

### 📊 Přehled statistik
- Celkový počet návštěvníků
- Průměrná denní návštěvnost
- Den s nejvyšší návštěvností
- Měsíční trendy

### 📈 Vizualizace dat
- Graf návštěvnosti za poslední měsíc
- Srovnání skutečné vs. predikované návštěvnosti
- Interaktivní grafy pomocí Chart.js

### 🔮 Predikce
1. **Jednoduchá predikce** - Pro jeden konkrétní den
   - Výběr data
   - Označení státního svátku/prázdnin
   - Volba otevírací doby

2. **Rozsahová predikce** - Pro více dní najednou
   - Zadání rozsahu dat
   - Zobrazení v tabulce
   - Celková predikce pro období

### ⚙️ Nastavení
- Konfigurace API endpointu
- Informace o použitém modelu
- Status připojení k API

## Struktura projektu

```
frontend/
├── src/
│   ├── app/              # Next.js App Router stránky
│   ├── components/       # React komponenty
│   ├── lib/              # Utility funkce a API klient
│   └── types/            # TypeScript definice typů
├── public/               # Statické soubory
└── package.json          # Závislosti projektu
```

## Komponenty

- **Sidebar** - Navigační menu
- **Header** - Hlavička s indikátorem připojení
- **StatsCards** - Karty se statistikami
- **VisitorChart** - Graf návštěvnosti
- **PredictionForm** - Formulář pro jednoduchou predikci
- **RangePredictionForm** - Formulář pro rozsahovou predikci
- **HealthStatus** - Indikátor stavu API

## Technologie

- **Next.js 14** - React framework
- **TypeScript** - Typová bezpečnost
- **Tailwind CSS** - Styling
- **Headless UI** - Přístupné UI komponenty
- **Chart.js** - Grafy a vizualizace
- **date-fns** - Práce s datumy

## Řešení problémů

### API není dostupné
1. Zkontrolujte, že backend běží na http://localhost:5000
2. Ověřte nastavení v `.env.local`
3. Zkontrolujte konzoli prohlížeče pro chyby CORS

### Závislosti se nenainstalují
```bash
# Vyčistěte npm cache
npm cache clean --force

# Smažte node_modules a lock soubor
rm -rf node_modules package-lock.json

# Reinstalujte
npm install
```

### Port 3000 je obsazený
```bash
# Použijte jiný port
PORT=3001 npm run dev
```

## Deployment

### Vercel (doporučeno pro Next.js)
```bash
npm install -g vercel
vercel
```

### Manuální build
```bash
npm run build
npm start
```

## Podpora

Pro dotazy a podporu kontaktujte tým Techmania.
