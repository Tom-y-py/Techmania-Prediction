"use client";

import { useState } from "react";
import { format } from "date-fns";
import { cs } from "date-fns/locale";
import { api } from "@/lib/api";
import type { RangePredictionResponse } from "@/types/api";
import ExportButton from "./ExportButton";
import { useTranslations } from "@/lib/i18n";

export default function RangePredictionForm() {
  const t = useTranslations('predictions');
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<RangePredictionResponse | null>(null);
  const [error, setError] = useState("");

  const isClosed = (date: string, dayOfWeek: string): boolean => {
    if (dayOfWeek !== "Pondělí") {
      return false;
    }
    
    const dateObj = new Date(date);
    const month = dateObj.getMonth() + 1;
    const isSummer = month >= 6 && month <= 8;
    
    return !isSummer;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const prediction = await api.predictRange({
        start_date: startDate,
        end_date: endDate,
      });
      setResult(prediction);
    } catch (err: any) {
      const errorMessage =
        err.response?.data?.detail ||
        t('error') ||
        "Chyba při načítání predikce";
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white dark:bg-gray-800 shadow-sm ring-1 ring-gray-900/5 dark:ring-gray-700 sm:rounded-xl">
      <div className="px-4 py-6 sm:p-8">
        <div className="w-full">
          <h2 className="text-base font-semibold leading-7 text-gray-900 dark:text-white">
            {t('rangeTitle')}
          </h2>
          <p className="mt-1 text-sm leading-6 text-gray-600 dark:text-gray-400">
            {t('rangeDescription')}
          </p>

          <form
            onSubmit={handleSubmit}
            className="mt-8 grid grid-cols-1 gap-6 sm:grid-cols-2"
          >
            <div>
              <label
                htmlFor="startDate"
                className="block text-sm font-medium leading-6 text-gray-900 dark:text-gray-300"
              >
                {t('dateFrom')}
              </label>
              <input
                type="date"
                id="startDate"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                required
                className="mt-2 block w-full rounded-md border-0 py-1.5 px-3 text-gray-900 dark:text-white dark:bg-gray-700 shadow-sm ring-1 ring-inset ring-gray-300 dark:ring-gray-600 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-techmania-blue sm:text-sm sm:leading-6"
              />
            </div>

            <div>
              <label
                htmlFor="endDate"
                className="block text-sm font-medium leading-6 text-gray-900 dark:text-gray-300"
              >
                {t('dateTo')}
              </label>
              <input
                type="date"
                id="endDate"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
                required
                className="mt-2 block w-full rounded-md border-0 py-1.5 px-3 text-gray-900 dark:text-white dark:bg-gray-700 shadow-sm ring-1 ring-inset ring-gray-300 dark:ring-gray-600 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-techmania-blue sm:text-sm sm:leading-6"
              />
            </div>

            <div className="sm:col-span-2">
              <button
                type="submit"
                disabled={loading}
                className="w-full rounded-md bg-techmania-blue px-3.5 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-blue-700 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-techmania-blue disabled:opacity-50"
              >
                {loading ? t('loading') : t('submit')}
              </button>
            </div>
          </form>

          {error && (
            <div className="mt-6 rounded-md bg-red-50 p-4">
              <p className="text-sm text-red-800">{error}</p>
            </div>
          )}

          {result && (
            <div className="mt-6">
              <div className="rounded-lg bg-gradient-to-r from-blue-50 to-green-50 dark:from-blue-900/30 dark:to-green-900/30 p-6 mb-6">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                      {t('totalPrediction')}
                    </h3>
                    <p className="text-3xl font-bold text-techmania-blue dark:text-blue-400">
                      {Math.round(result.total_predicted).toLocaleString(
                        "cs-CZ"
                      )}{" "}
                      {t('visitors')}
                    </p>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                      {t('forPeriod')} {result.predictions.length} {t('days')}
                    </p>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      {t('averageDaily')}{" "}
                      {Math.round(result.average_daily).toLocaleString("cs-CZ")}{" "}
                      {t('perDay')}
                    </p>
                  </div>
                  <ExportButton
                    data={result.predictions}
                    filename={`techmania_predikce_${startDate}_${endDate}`}
                  />
                </div>
              </div>

              <div className="overflow-hidden shadow ring-1 ring-black dark:ring-gray-700 ring-opacity-5 sm:rounded-lg">
                <table className="min-w-full divide-y divide-gray-300 dark:divide-gray-700">
                  <thead className="bg-gray-50 dark:bg-gray-900">
                    <tr>
                      <th className="py-3.5 pl-4 pr-3 text-left text-sm font-semibold text-gray-900 dark:text-gray-100 sm:pl-6">
                        {t('date')}
                      </th>
                      <th className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900 dark:text-gray-100">
                        {t('day')}
                      </th>
                      <th className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900 dark:text-gray-100">
                        {t('visitorsShort')}
                      </th>
                      <th className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900 dark:text-gray-100">
                        {t('interval')}
                      </th>
                      <th className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900 dark:text-gray-100">
                        {t('weather')}
                      </th>
                      <th className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900 dark:text-gray-100">
                        {t('holiday')}
                      </th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-200 dark:divide-gray-700 bg-white dark:bg-gray-800">
                    {result.predictions.map((prediction, idx) => (
                      <tr
                        key={idx}
                        className={prediction.is_weekend ? "bg-blue-50 dark:bg-blue-900/20" : ""}
                      >
                        <td className="whitespace-nowrap py-4 pl-4 pr-3 text-sm font-medium text-gray-900 dark:text-gray-100 sm:pl-6">
                          {format(new Date(prediction.date), "dd.MM.yyyy", {
                            locale: cs,
                          })}
                        </td>
                        <td className="whitespace-nowrap px-3 py-4 text-sm text-gray-700 dark:text-gray-300">
                          <div className="flex items-center gap-2">
                            {prediction.day_of_week}
                            {prediction.is_weekend && (
                              <span className="inline-flex items-center rounded-full bg-blue-100 px-2 py-0.5 text-xs font-medium text-blue-800">
                                {t('weekend')}
                              </span>
                            )}
                            {isClosed(prediction.date, prediction.day_of_week) && (
                              <span className="inline-flex items-center rounded-full bg-red-100 px-2 py-0.5 text-xs font-medium text-red-800">
                                {t('closed')}
                              </span>
                            )}
                          </div>
                        </td>
                        <td className="whitespace-nowrap px-3 py-4 text-sm font-bold text-gray-900 dark:text-gray-100">
                          {Math.round(
                            prediction.predicted_visitors
                          ).toLocaleString("cs-CZ")}
                        </td>
                        <td className="whitespace-nowrap px-3 py-4 text-xs text-gray-500 dark:text-gray-400">
                          {Math.round(
                            prediction.confidence_interval.lower
                          ).toLocaleString("cs-CZ")}{" "}
                          -{" "}
                          {Math.round(
                            prediction.confidence_interval.upper
                          ).toLocaleString("cs-CZ")}
                        </td>
                        <td className="px-3 py-4 text-sm text-gray-700 dark:text-gray-300">
                          <div className="flex flex-col gap-1">
                            <div className="flex items-center gap-2">
                              <span className="font-medium">
                                {prediction.weather_info.temperature_mean.toFixed(
                                  1
                                )}
                                °C
                              </span>
                              {prediction.weather_info.is_nice_weather && (
                                <span>☀️</span>
                              )}
                            </div>
                            <div className="text-xs text-gray-500">
                              {prediction.weather_info.precipitation > 0 && (
                                <span>
                                  🌧️{" "}
                                  {prediction.weather_info.precipitation.toFixed(
                                    1
                                  )}
                                  mm
                                </span>
                              )}
                              {prediction.weather_info.precipitation === 0 && (
                                <span>{t('noPrecipitation')}</span>
                              )}
                            </div>
                            <div className="text-xs text-gray-400">
                              {prediction.weather_info.weather_description}
                            </div>
                          </div>
                        </td>
                        <td className="px-3 py-4 text-sm">
                          {prediction.holiday_info.is_holiday ? (
                            <div className="flex flex-col gap-1">
                              <span className="inline-flex items-center rounded-full bg-amber-100 px-2 py-1 text-xs font-medium text-amber-800">
                                📅 {t('holiday')}
                              </span>
                              {prediction.holiday_info.holiday_name && (
                                <span className="text-xs text-gray-600">
                                  {prediction.holiday_info.holiday_name}
                                </span>
                              )}
                            </div>
                          ) : (
                            <span className="text-xs text-gray-400">—</span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
