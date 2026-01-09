'use client';

import { useEffect, useRef } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

export default function VisitorChart() {
  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
      },
      title: {
        display: true,
        text: 'Návštěvnost za poslední měsíc',
        font: {
          size: 16,
          weight: 'bold',
        },
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        ticks: {
          callback: function(value: any) {
            return value.toLocaleString('cs-CZ');
          }
        }
      }
    }
  };

  const labels = [
    '1.12', '2.12', '3.12', '4.12', '5.12', '6.12', '7.12',
    '8.12', '9.12', '10.12', '11.12', '12.12', '13.12', '14.12',
    '15.12', '16.12', '17.12', '18.12', '19.12', '20.12', '21.12',
    '22.12', '23.12', '24.12', '25.12', '26.12', '27.12', '28.12',
    '29.12', '30.12', '31.12'
  ];

  const data = {
    labels,
    datasets: [
      {
        label: 'Skutečná návštěvnost',
        data: [
          3200, 3100, 3400, 3500, 3800, 4200, 4100,
          3300, 3200, 3600, 3700, 4000, 4500, 4300,
          5200, 5400, 5100, 3400, 3300, 3700, 3900,
          4200, 4800, 4500, 6200, 6500, 6800, 5200,
          5100, 5400, 7200
        ],
        borderColor: 'rgb(0, 102, 204)',
        backgroundColor: 'rgba(0, 102, 204, 0.1)',
        fill: true,
        tension: 0.4,
      },
      {
        label: 'Predikovaná návštěvnost',
        data: [
          3100, 3050, 3350, 3450, 3750, 4150, 4050,
          3250, 3150, 3550, 3650, 3950, 4450, 4250,
          5150, 5350, 5050, 3350, 3250, 3650, 3850,
          4150, 4750, 4450, 6150, 6450, 6750, 5150,
          5050, 5350, 7150
        ],
        borderColor: 'rgb(0, 204, 102)',
        backgroundColor: 'rgba(0, 204, 102, 0.1)',
        fill: true,
        tension: 0.4,
        borderDash: [5, 5],
      },
    ],
  };

  return (
    <div className="bg-white shadow-sm ring-1 ring-gray-900/5 sm:rounded-xl p-6">
      <div style={{ height: '400px' }}>
        <Line options={options} data={data} />
      </div>
    </div>
  );
}
