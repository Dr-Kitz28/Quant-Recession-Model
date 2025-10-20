import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'

const apiBase = import.meta.env.VITE_API_BASE || ''
console.log('Environment VITE_API_BASE:', import.meta.env.VITE_API_BASE)
console.log('Using API Base:', apiBase)

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App apiBase={apiBase} />
  </StrictMode>,
)
