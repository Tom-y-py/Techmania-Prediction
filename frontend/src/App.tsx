import { PredictionForm } from './components/PredictionForm'

function App() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-primary-500 via-purple-500 to-secondary-500 p-6">
      <div className="max-w-6xl mx-auto">
        <div className="bg-white rounded-3xl shadow-2xl p-10">
          <div className="text-center mb-10">
            <h1 className="text-5xl font-bold bg-gradient-to-r from-primary-500 to-secondary-500 bg-clip-text text-transparent mb-3">
              🔬 Techmania
            </h1>
            <p className="text-gray-600 text-lg">
              Predikce návštěvnosti pomocí AI
            </p>
          </div>
          
          <PredictionForm />
        </div>
      </div>
    </div>
  )
}

export default App

