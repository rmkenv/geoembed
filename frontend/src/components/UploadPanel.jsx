import React, { useState } from 'react'

export default function UploadPanel({ apiBase }) {
  const [file, setFile] = useState(null)
  const [status, setStatus] = useState(null)

  async function uploadFile() {
    if (!file) return
    const reader = new FileReader()
    reader.onload = async () => {
      try {
        const features = JSON.parse(reader.result).features
        const res = await fetch(`${apiBase}/embed/vector/`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ features })
        })
        if (res.ok) {
          setStatus('Upload and embedding succeeded.')
        } else {
          setStatus('Upload failed.')
        }
      } catch {
        setStatus('Invalid GeoJSON file.')
      }
    }
    reader.readAsText(file)
  }

  return (
    <div style={{ marginTop: 40 }}>
      <h2>Upload GeoJSON for Embedding</h2>
      <input
        type="file"
        accept=".json,.geojson"
        onChange={e => setFile(e.target.files[0])}
      />
      <button onClick={uploadFile} disabled={!file} style={{ marginLeft: 10 }}>
        Upload & Embed
      </button>
      {status && <p>{status}</p>}
    </div>
  )
}
