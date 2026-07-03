import axios from 'axios'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || '',
  timeout: 15000,
  headers: { 'Content-Type': 'application/json' },
})

api.interceptors.response.use(
  (res) => res,
  (error) => {
    const detail = error.response?.data?.detail || error.message
    console.error(`API error [${error.config?.method?.toUpperCase()} ${error.config?.url}]:`, detail)
    return Promise.reject(error)
  }
)

export function getImageUrl(url) {
  if (!url) return null
  if (url.startsWith('http://') || url.startsWith('https://') || url.startsWith('data:')) return url
  return `/${url}`
}

export default api
