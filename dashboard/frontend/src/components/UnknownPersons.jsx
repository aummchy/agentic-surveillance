import React, { useState, useEffect } from 'react'
import api, { getImageUrl } from '../utils/api'

const NEW_PERSON_WINDOW_MS = 30000

function UnknownPersons({ onVerify, refreshKey }) {
  const [unknowns, setUnknowns] = useState([])
  const [loading, setLoading] = useState(true)
  const [seenIds, setSeenIds] = useState(() => {
    try { return new Set(JSON.parse(sessionStorage.getItem('seenUnknowns') || '[]')) }
    catch { return new Set() }
  })

  useEffect(() => {
    fetchPersons()
  }, [refreshKey])

  const fetchPersons = async () => {
    setLoading(true)
    try {
      const response = await api.get('/api/faces', {
        params: { status: 'unknown', limit: 50 }
      })
      const faces = response.data.faces || []
      setUnknowns(faces)
    } catch (error) {
      console.error('Failed to fetch persons:', error)
      setUnknowns([])
    } finally {
      setLoading(false)
    }
  }

  const formatTime = (dateString) => {
    if (!dateString) return 'Unknown'
    const date = new Date(dateString)
    return date.toLocaleString()
  }

  const formatTimeShort = (dateString) => {
    if (!dateString) return ''
    const date = new Date(dateString)
    const now = new Date()
    const diffMs = now - date
    const diffMins = Math.floor(diffMs / 60000)
    if (diffMins < 1) return 'Just now'
    if (diffMins < 60) return `${diffMins}m ago`
    const diffHrs = Math.floor(diffMins / 60)
    if (diffHrs < 24) return `${diffHrs}h ago`
    return `${Math.floor(diffHrs / 24)}d ago`
  }

  const getLatestImage = (person) => {
    if (person.images && person.images.length > 0) {
      return getImageUrl(person.images[person.images.length - 1].url)
    }
    return null
  }

  const getLastSeen = (person) => {
    if (person.images && person.images.length > 0) {
      const lastImg = person.images[person.images.length - 1]
      return lastImg.captured_at || person.updated_at
    }
    return person.updated_at || person.created_at
  }

  const isNewPerson = (person) => {
    if (seenIds.has(person.person_id)) return false
    const created = new Date(person.created_at).getTime()
    return Date.now() - created < NEW_PERSON_WINDOW_MS
  }

  const markAsSeen = (personId) => {
    setSeenIds((prev) => {
      const next = new Set(prev)
      next.add(personId)
      try { sessionStorage.setItem('seenUnknowns', JSON.stringify([...next])) } catch {}
      return next
    })
  }

  return (
    <div>
      <div className="panel-header">
        <span className="panel-title">
          Unknown Persons
        </span>
        <span className="panel-count">{unknowns.length} persons</span>
      </div>

      {loading ? (
        <div className="loading">Loading...</div>
      ) : unknowns.length === 0 ? (
        <div className="empty-state">
          No unknown persons detected
        </div>
      ) : (
        <div className="persons-grid">
          {unknowns.map((person) => (
            <div
              key={person.person_id}
              className={`person-card ${isNewPerson(person) ? 'person-card-new' : ''}`}
              onMouseEnter={() => markAsSeen(person.person_id)}
              onClick={() => markAsSeen(person.person_id)}
            >
              {getLatestImage(person) ? (
                <img
                  src={getLatestImage(person)}
                  alt={person.name || 'Unknown'}
                  className="person-image"
                />
              ) : (
                <div className="person-image" style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: '#666'
                }}>
                  No Image
                </div>
              )}
              <div className="person-info">
                <div className="person-name">
                  {person.verified ? person.name : 'Unknown'}
                </div>
                <div className="person-id">ID: {person.person_id.split('_').slice(-1)[0]}</div>
                {person.images && person.images[0] && person.images[0].id && (
                  <div className="person-photo-id" style={{fontSize: '0.7em', color: '#888'}}>
                    Photo: {person.images[0].id.slice(0, 8)}
                  </div>
                )}
                <div className="person-time-row">
                  <span className="person-time">First: {formatTimeShort(person.created_at)}</span>
                  <span className="person-time-sep">|</span>
                  <span className="person-time">Last: {formatTimeShort(getLastSeen(person))}</span>
                </div>
                {person.images && person.images.length > 1 && (
                  <div className="person-image-count">{person.images.length} images captured</div>
                )}
                <button
                  className="verify-btn"
                  onClick={() => onVerify(person)}
                >
                  Verify Person
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default UnknownPersons
