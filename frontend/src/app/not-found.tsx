export default function NotFound() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-gray-50 px-4">
      <div className="max-w-md text-center">
        <div className="mb-8">
          <h1 className="text-9xl font-bold text-techmania-blue">404</h1>
        </div>
        <h2 className="mb-4 text-2xl font-semibold text-gray-900">
          Stránka nenalezena
        </h2>
        <p className="mb-8 text-gray-600">
          Omlouváme se, ale stránka, kterou hledáte, neexistuje.
        </p>
        <a
          href="/"
          className="rounded-md bg-techmania-blue px-4 py-2 text-sm font-semibold text-white shadow-sm hover:bg-blue-700"
        >
          Zpět na hlavní stránku
        </a>
      </div>
    </div>
  );
}
