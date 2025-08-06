import React, { useEffect, useState } from 'react'
import MapView from './components/MapView'
import SearchBox from './components/SearchBox'
import UploadPanel from './components/UploadPanel'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export default function App() {
  const [cities, setCities] = useState([])
  const [searchResults, setSearchResults] = useState([])
  const [loading, setLoading] = useState(false)

  // Fetch sample cities on mount
  useEffect(() => {
    fetch(`${API_BASE}/embed/vector/models`)
      .catch(() => {}) // Ignore errors from models
    fetch('/data/sample_cities.geojson')
      .then(res => res.json())
      .then(data => setCities(data.features || []))
      .catch(() => {})
  }, [])

  async function doSearch(query, topK = 5) {
    if (!query) return setSearchResults([])
    setLoading(true)
    try {
      const res = await fetch(`${API_BASE}/search/semantic`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query_text: query, k: topK })
      })
      if (res.ok) {
        const data = await res.json()
        setSearchResults(data)
      } else {
        setSearchResults([])
      }
    } catch {
      setSearchResults([])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ maxWidth: 900, margin: '20px auto', fontFamily: 'Arial, sans-serif' }}>
      <h1>Geospatial Embeddings Platform Demo</h1>

      <SearchBox onSearch={doSearch} loading={loading} />

      <SearchResults results={searchResults} />

      <MapView cities={cities} searchResults={searchResults} />

      <UploadPanel apiBase={API_BASE} />
    </div>
  )
}

function SearchResults({ results }) {
  if (!results || results.length === 0) return null

  return (
    <div style={{ marginTop: 20 }}>
      <h2>Search Results</h2>
      {results.map(r => (
        <div key={r.id} style={{ padding: 8, borderBottom: '1px solid #ddd' }}>
          <strong>{r.name}</strong> ({(r.similarity * 100).toFixed(1)}% similarity)
          <div>{r.properties?.description || ''}</div>
        </div>
      ))}
    </div>
  )
}
