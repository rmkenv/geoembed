import React, { useEffect, useRef } from 'react'
import L from 'leaflet'

export default function MapView({ cities = [], searchResults = [] }) {
  const mapRef = useRef(null)

  useEffect(() => {
    if (!mapRef.current) {
      mapRef.current = L.map('map').setView([20, 0], 2)
      L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(mapRef.current)
    }
  }, [])

  useEffect(() => {
    if (!mapRef.current) return
    mapRef.current.eachLayer(layer => {
      if (layer.options && layer.options.pane === 'markerPane') {
        mapRef.current.removeLayer(layer)
      }
    })

    cities.forEach(f => {
      const [lon, lat] = f.geometry.coordinates
      L.marker([lat, lon])
        .addTo(mapRef.current)
        .bindPopup(`<b>${f.properties.name}</b><br>${f.properties.description || ''}`)
    })

    searchResults.forEach(f => {
      const geom = f.geometry || f.geometry_json
      if (!geom || geom.type !== 'Point') return
      const [lon, lat] = geom.coordinates || geom.coordinates
      L.circleMarker([lat, lon], { color: 'red', radius: 8, opacity: 0.7 })
        .addTo(mapRef.current)
        .bindPopup(`<b>${f.name}</b><br>Similarity: ${(f.similarity*100).toFixed(1)}%`)
    })
  }, [cities, searchResults])

  return <div id="map" style={{ height: 400, width: '100%', marginTop: 20 }} />
}
