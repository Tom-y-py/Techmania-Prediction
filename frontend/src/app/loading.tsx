export default function Loading() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-gray-50">
      <div className="text-center">
        <div className="inline-block h-12 w-12 animate-spin rounded-full border-4 border-solid border-techmania-blue border-r-transparent"></div>
        <p className="mt-4 text-sm text-gray-600">Načítání...</p>
      </div>
    </div>
  );
}
