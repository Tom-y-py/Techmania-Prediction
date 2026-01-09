'use client';

import { useEffect } from 'react';

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error(error);
  }, [error]);

  return (
    <div className="flex min-h-screen items-center justify-center bg-gray-50 px-4">
      <div className="max-w-md text-center">
        <div className="mb-8">
          <h1 className="text-6xl font-bold text-techmania-blue">Chyba</h1>
        </div>
        <h2 className="mb-4 text-2xl font-semibold text-gray-900">
          Něco se pokazilo
        </h2>
        <p className="mb-8 text-gray-600">
          Omlouváme se, ale došlo k neočekávané chybě. Zkuste to prosím znovu.
        </p>
        <div className="space-x-4">
          <button
            onClick={reset}
            className="rounded-md bg-techmania-blue px-4 py-2 text-sm font-semibold text-white shadow-sm hover:bg-blue-700"
          >
            Zkusit znovu
          </button>
          <a
            href="/"
            className="rounded-md bg-gray-200 px-4 py-2 text-sm font-semibold text-gray-900 shadow-sm hover:bg-gray-300"
          >
            Domů
          </a>
        </div>
      </div>
    </div>
  );
}
