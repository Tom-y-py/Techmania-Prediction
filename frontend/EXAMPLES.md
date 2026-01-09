# Příklady použití API

## Základní použití API klienta

### Import
```typescript
import { api } from '@/lib/api';
```

## Jednoduchá predikce

### Příklad 1: Běžný pracovní den
```typescript
const result = await api.predict({
  date: '2026-01-15',
  is_holiday: false,
  opening_hours: '09:00-17:00'
});

console.log(result);
// {
//   date: '2026-01-15',
//   predicted_visitors: 3421,
//   confidence_interval: {
//     lower: 3100,
//     upper: 3750
//   }
// }
```

### Příklad 2: Víkendový den
```typescript
const result = await api.predict({
  date: '2026-01-17', // Sobota
  is_holiday: false,
  opening_hours: '10:00-18:00'
});
```

### Příklad 3: Státní svátek
```typescript
const result = await api.predict({
  date: '2026-01-01', // Nový rok
  is_holiday: true,
  opening_hours: '10:00-18:00'
});
```

### Příklad 4: Prodloužená otevírací doba
```typescript
const result = await api.predict({
  date: '2026-07-15',
  is_holiday: false,
  opening_hours: '09:00-19:00' // Letní provoz
});
```

## Rozsahová predikce

### Příklad 1: Týdenní predikce
```typescript
const result = await api.predictRange({
  start_date: '2026-01-15',
  end_date: '2026-01-21'
});

console.log(result);
// {
//   predictions: [
//     { date: '2026-01-15', predicted_visitors: 3421 },
//     { date: '2026-01-16', predicted_visitors: 3542 },
//     { date: '2026-01-17', predicted_visitors: 4823 },
//     { date: '2026-01-18', predicted_visitors: 5124 },
//     { date: '2026-01-19', predicted_visitors: 3345 },
//     { date: '2026-01-20', predicted_visitors: 3256 },
//     { date: '2026-01-21', predicted_visitors: 3189 }
//   ],
//   total_predicted: 26700
// }
```

### Příklad 2: Měsíční predikce
```typescript
const result = await api.predictRange({
  start_date: '2026-02-01',
  end_date: '2026-02-28'
});

// Celkový počet návštěvníků za únor
console.log(result.total_predicted);
```

### Příklad 3: Vánoční období
```typescript
const result = await api.predictRange({
  start_date: '2026-12-20',
  end_date: '2026-12-31'
});
```

## Health Check

### Příklad 1: Kontrola stavu API
```typescript
const health = await api.healthCheck();

console.log(health);
// {
//   status: 'healthy',
//   model_loaded: true
// }

if (health.status === 'healthy' && health.model_loaded) {
  console.log('API je připraveno k použití');
}
```

## Error Handling

### Příklad 1: Try-catch
```typescript
try {
  const result = await api.predict({
    date: '2026-01-15',
    is_holiday: false,
    opening_hours: '09:00-17:00'
  });
  console.log('Predikce úspěšná:', result);
} catch (error) {
  console.error('Chyba při predikci:', error.message);
}
```

### Příklad 2: Validace vstupu
```typescript
const date = '2026-01-15';
const isValidDate = !isNaN(Date.parse(date));

if (!isValidDate) {
  console.error('Neplatné datum');
  return;
}

try {
  const result = await api.predict({
    date,
    is_holiday: false,
    opening_hours: '09:00-17:00'
  });
} catch (error) {
  console.error(error);
}
```

## Použití v React komponentě

### Příklad 1: useState a useEffect
```typescript
'use client';

import { useState, useEffect } from 'react';
import { api } from '@/lib/api';

export default function PredictionComponent() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const fetchPrediction = async () => {
    setLoading(true);
    setError('');
    
    try {
      const data = await api.predict({
        date: '2026-01-15',
        is_holiday: false,
        opening_hours: '09:00-17:00'
      });
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPrediction();
  }, []);

  if (loading) return <div>Načítání...</div>;
  if (error) return <div>Chyba: {error}</div>;
  if (!result) return null;

  return (
    <div>
      <h2>Predikce pro {result.date}</h2>
      <p>Počet návštěvníků: {result.predicted_visitors}</p>
    </div>
  );
}
```

### Příklad 2: Form submission
```typescript
'use client';

import { useState } from 'react';
import { api } from '@/lib/api';

export default function PredictionForm() {
  const [formData, setFormData] = useState({
    date: '',
    is_holiday: false,
    opening_hours: '09:00-17:00'
  });
  const [result, setResult] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    const prediction = await api.predict(formData);
    setResult(prediction);
  };

  return (
    <form onSubmit={handleSubmit}>
      <input
        type="date"
        value={formData.date}
        onChange={(e) => setFormData({
          ...formData,
          date: e.target.value
        })}
      />
      
      <label>
        <input
          type="checkbox"
          checked={formData.is_holiday}
          onChange={(e) => setFormData({
            ...formData,
            is_holiday: e.target.checked
          })}
        />
        Státní svátek
      </label>
      
      <select
        value={formData.opening_hours}
        onChange={(e) => setFormData({
          ...formData,
          opening_hours: e.target.value
        })}
      >
        <option value="09:00-17:00">09:00 - 17:00</option>
        <option value="09:00-18:00">09:00 - 18:00</option>
        <option value="10:00-18:00">10:00 - 18:00</option>
      </select>
      
      <button type="submit">Spustit predikci</button>
      
      {result && (
        <div>
          <h3>Výsledek</h3>
          <p>Návštěvníků: {result.predicted_visitors}</p>
          <p>
            Interval: {result.confidence_interval.lower} - 
            {result.confidence_interval.upper}
          </p>
        </div>
      )}
    </form>
  );
}
```

## Export dat

### Příklad 1: Export CSV
```typescript
const exportToCSV = (predictions) => {
  const headers = ['Datum', 'Počet návštěvníků'];
  const rows = predictions.map(p => 
    [p.date, p.predicted_visitors].join(',')
  );
  
  const csv = [headers.join(','), ...rows].join('\n');
  const blob = new Blob(['\ufeff' + csv], { 
    type: 'text/csv;charset=utf-8;' 
  });
  
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = 'predikce.csv';
  link.click();
};

// Použití
const result = await api.predictRange({
  start_date: '2026-01-15',
  end_date: '2026-01-21'
});

exportToCSV(result.predictions);
```

### Příklad 2: Export JSON
```typescript
const exportToJSON = (data) => {
  const json = JSON.stringify(data, null, 2);
  const blob = new Blob([json], { type: 'application/json' });
  
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = 'predikce.json';
  link.click();
};

// Použití
const result = await api.predictRange({
  start_date: '2026-01-15',
  end_date: '2026-01-21'
});

exportToJSON(result);
```

## Pokročilé použití

### Příklad 1: Batch predikce s Promise.all
```typescript
const dates = [
  '2026-01-15',
  '2026-01-16',
  '2026-01-17'
];

const predictions = await Promise.all(
  dates.map(date => 
    api.predict({
      date,
      is_holiday: false,
      opening_hours: '09:00-17:00'
    })
  )
);

console.log(predictions);
```

### Příklad 2: Retry logika
```typescript
async function predictWithRetry(data, maxRetries = 3) {
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await api.predict(data);
    } catch (error) {
      if (i === maxRetries - 1) throw error;
      await new Promise(resolve => 
        setTimeout(resolve, 1000 * (i + 1))
      );
    }
  }
}

// Použití
const result = await predictWithRetry({
  date: '2026-01-15',
  is_holiday: false,
  opening_hours: '09:00-17:00'
});
```

### Příklad 3: Caching
```typescript
const cache = new Map();

async function getCachedPrediction(data) {
  const key = JSON.stringify(data);
  
  if (cache.has(key)) {
    console.log('Vráceno z cache');
    return cache.get(key);
  }
  
  const result = await api.predict(data);
  cache.set(key, result);
  
  return result;
}
```

## Formátování výsledků

### Příklad 1: Formátování čísla
```typescript
const result = await api.predict({
  date: '2026-01-15',
  is_holiday: false,
  opening_hours: '09:00-17:00'
});

const formatted = Math.round(result.predicted_visitors)
  .toLocaleString('cs-CZ');

console.log(`Očekávaný počet: ${formatted} návštěvníků`);
```

### Příklad 2: Formátování data
```typescript
import { format } from 'date-fns';
import { cs } from 'date-fns/locale';

const result = await api.predict({
  date: '2026-01-15',
  is_holiday: false,
  opening_hours: '09:00-17:00'
});

const formattedDate = format(
  new Date(result.date), 
  'PPPP', 
  { locale: cs }
);

console.log(formattedDate); 
// "čtvrtek 15. ledna 2026"
```
