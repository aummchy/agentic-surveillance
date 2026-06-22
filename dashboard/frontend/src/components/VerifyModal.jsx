import React, { useState } from 'react'
import axios from 'axios'

function VerifyModal({ person, onVerified, onClose }) {
  const [name, setName] = useState('')
  const [alertLevel, setAlertLevel] = useState('low')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const handleSubmit = async (e) => {
    e.preventDefault()
    
    if (!name.trim()) {
      setError('Please enter a name')
      return
    }

    setLoading(true)
    setError('')

    try {
      await axios.post(`/api/faces/${person.person_id}/verify`, {
        name: name.trim(),
        alert_level: alertLevel
      })
      onVerified()
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to verify person')
    } finally {
      setLoading(false)
    }
  }

  const getPersonImage = () => {
    if (person.images && person.images.length > 0) {
      return person.images[0].url
    }
    return null
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <h2>Verify Person</h2>
        
        <div style={{ 
          display: 'flex', 
          alignItems: 'center', 
          gap: '1rem',
          marginBottom: '1.5rem',
          padding: '1rem',
          background: '#0f3460',
          borderRadius: '8px'
        }}>
          {getPersonImage() ? (
            <img
              src={getPersonImage()}
              alt="Person"
              style={{
                width: '80px',
                height: '80px',
                borderRadius: '8px',
                objectFit: 'cover'
              }}
            />
          ) : (
            <div style={{
              width: '80px',
              height: '80px',
              borderRadius: '8px',
              background: '#1a1a2e',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: '#666'
            }}>
              No Image
            </div>
          )}
          <div>
            <div style={{ fontWeight: '600', marginBottom: '0.25rem' }}>
              Track ID: {person.person_id}
            </div>
            <div style={{ fontSize: '0.85rem', color: '#888' }}>
              First detected: {new Date(person.created_at).toLocaleString()}
            </div>
          </div>
        </div>

        <form className="modal-form" onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="name">Person's Name</label>
            <input
              type="text"
              id="name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Enter name..."
              disabled={loading}
            />
          </div>

          <div className="form-group">
            <label htmlFor="alertLevel">Alert Level</label>
            <select
              id="alertLevel"
              value={alertLevel}
              onChange={(e) => setAlertLevel(e.target.value)}
              disabled={loading}
            >
              <option value="none">None (Silent - no alerts)</option>
              <option value="low">Low (Log only)</option>
              <option value="medium">Medium (Standard alert)</option>
              <option value="high">High (Priority alert)</option>
            </select>
          </div>

          {error && (
            <div style={{
              padding: '0.75rem',
              background: '#ef4444',
              borderRadius: '6px',
              color: '#fff',
              fontSize: '0.9rem'
            }}>
              {error}
            </div>
          )}

          <div className="modal-actions">
            <button
              type="button"
              className="btn btn-secondary"
              onClick={onClose}
              disabled={loading}
            >
              Cancel
            </button>
            <button
              type="submit"
              className="btn btn-primary"
              disabled={loading}
            >
              {loading ? 'Verifying...' : 'Verify & Save'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

export default VerifyModal
