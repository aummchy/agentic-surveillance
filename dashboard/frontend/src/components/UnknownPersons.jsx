import React, { useState, useEffect } from 'react'
import axios from 'axios'

function UnknownPersons({ onVerify, refreshKey, onUnknownsLoaded }) {
  const [unknowns, setUnknowns] = useState([])
  const [loading, setLoading] = useState(true)
  const [activeTab, setActiveTab] = useState('unknown')

  useEffect(() => {
    fetchPersons()
  }, [refreshKey, activeTab])

  const fetchPersons = async () => {
    setLoading(true)
    try {
      const response = await axios.get(`/api/faces`, {
        params: { status: activeTab, limit: 50 }
      })
      const faces = response.data.faces || []
      setUnknowns(faces)
      if (onUnknownsLoaded) {
        onUnknownsLoaded(faces)
      }
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

  const getImageUrl = (url) => {
    if (!url) return null
    if (url.startsWith('http://') || url.startsWith('https://') || url.startsWith('data:')) return url
    return `http://localhost:8000/${url}`
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
    const fiveSecsAgo = Date.now() - 5000
    const created = new Date(person.created_at).getTime()
    return created > fiveSecsAgo
  }

  return (
    <div>
      <div className="panel-header">
        <span className="panel-title">
          {activeTab === 'unknown' ? 'Unknown Persons' : 'Verified Persons'}
        </span>
        <span className="panel-count">{unknowns.length} persons</span>
      </div>

      <div className="tabs">
        <button
          className={`tab ${activeTab === 'unknown' ? 'active' : ''}`}
          onClick={() => setActiveTab('unknown')}
        >
          Unknown
        </button>
        <button
          className={`tab ${activeTab === 'verified' ? 'active' : ''}`}
          onClick={() => setActiveTab('verified')}
        >
          Verified
        </button>
      </div>

      {loading ? (
        <div className="loading">Loading...</div>
      ) : unknowns.length === 0 ? (
        <div className="empty-state">
          {activeTab === 'unknown' 
            ? 'No unknown persons detected'
            : 'No verified persons yet'
          }
        </div>
      ) : (
        <div className="persons-grid">
          {unknowns.map((person) => (
            <div key={person.person_id} className={`person-card ${activeTab === 'unknown' && isNewPerson(person) ? 'person-card-new' : ''}`}>
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
                {!person.verified && (
                  <button
                    className="verify-btn"
                    onClick={() => onVerify(person)}
                  >
                    Verify Person
                  </button>
                )}
                {person.verified && (
                  <div className="verified-badge">
                    Verified - {person.alert_level} alert
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default UnknownPersons
