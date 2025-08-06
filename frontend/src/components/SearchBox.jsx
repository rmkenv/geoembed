import React, { useState } from 'react'

export default function SearchBox({ onSearch, loading }) {
  const [query, setQuery] = useState('')

  function handleSubmit(e) {
    e.preventDefault()
    onSearch(query)
  }

  return (
    <form onSubmit={handleSubmit} style={{ marginBottom: 10 }}>
      <input
        type="text"
        placeholder="Describe your search, e.g. 'large metropolitan city'"
        style={{ width: '75%', fontSize: 16, padding: 8 }}
        value={query}
        onChange={e => setQuery(e.target.value)}
      />
      <button type="submit" disabled={loading} style={{ padding: 8, marginLeft: 8 }}>
        {loading ? 'Searching...' : 'Search'}
      </button>
    </form>
  )
}
