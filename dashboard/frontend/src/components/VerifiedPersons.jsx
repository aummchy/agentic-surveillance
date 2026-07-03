import React, { useState, useEffect } from 'react'
import api, { getImageUrl } from '../utils/api'

function VerifiedPersons({ refreshKey }) {
  const [persons, setPersons] = useState([])
  const [loading, setLoading] = useState(true)
  const [deleting, setDeleting] = useState(null)
  const [confirmDelete, setConfirmDelete] = useState(null)

  useEffect(() => {
    fetchVerified()
  }, [refreshKey])

  const fetchVerified = async () => {
    setLoading(true)
    try {
      const response = await api.get('/api/faces', {
        params: { status: 'verified', limit: 50 }
      })
      setPersons(response.data.faces || [])
    } catch (error) {
      console.error('Failed to fetch verified persons:', error)
      setPersons([])
    } finally {
      setLoading(false)
    }
  }

  const handleDelete = async (personId) => {
    setDeleting(personId)
    try {
      await api.delete(`/api/faces/${personId}`)
      setPersons(prev => prev.filter(p => p.person_id !== personId))
      setConfirmDelete(null)
    } catch (error) {
      console.error('Failed to delete person:', error)
      alert('Failed to delete person. Please try again.')
    } finally {
      setDeleting(null)
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

  const getAlertColor = (level) => {
    switch (level) {
      case 'critical': return '#dc2626'
      case 'high': return '#ef4444'
      case 'medium': return '#f59e0b'
      case 'low': return '#3b82f6'
      default: return '#94a3b8'
    }
  }

  return (
    <div>
      <div className="panel-header">
        <span className="panel-title">
          Verified Persons
        </span>
        <span className="panel-count">{persons.length} persons</span>
      </div>

      {loading ? (
        <div className="loading">Loading...</div>
      ) : persons.length === 0 ? (
        <div className="empty-state">No verified persons yet</div>
      ) : (
        <div className="persons-grid">
          {persons.map((person) => (
            <div key={person.person_id} className="person-card verified-person-card">
              {getLatestImage(person) ? (
                <img
                  src={getLatestImage(person)}
                  alt={person.name || 'Verified'}
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
                <div className="person-name">{person.name || 'Unnamed'}</div>
                <div className="person-id">ID: {person.person_id.split('_').slice(-1)[0]}</div>
                <div className="person-time-row">
                  <span className="person-time">Verified: {formatTimeShort(person.verified_at)}</span>
                </div>
                {person.images && person.images.length > 1 && (
                  <div className="person-image-count">{person.images.length} images</div>
                )}
                <div className="verified-badge" style={{ background: getAlertColor(person.alert_level) }}>
                  {person.alert_level || 'none'} alert
                </div>
                {confirmDelete === person.person_id ? (
                  <div className="delete-confirm">
                    <span className="delete-confirm-text">Delete this person?</span>
                    <div className="delete-confirm-actions">
                      <button
                        className="delete-confirm-btn delete-confirm-yes"
                        onClick={() => handleDelete(person.person_id)}
                        disabled={deleting === person.person_id}
                      >
                        {deleting === person.person_id ? 'Deleting...' : 'Yes'}
                      </button>
                      <button
                        className="delete-confirm-btn delete-confirm-no"
                        onClick={() => setConfirmDelete(null)}
                      >
                        No
                      </button>
                    </div>
                  </div>
                ) : (
                  <button
                    className="delete-btn"
                    onClick={() => setConfirmDelete(person.person_id)}
                  >
                    Delete
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default VerifiedPersons
