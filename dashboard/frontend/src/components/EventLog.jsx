import React, { useState, useEffect, useRef, useCallback } from 'react'
import api, { getImageUrl } from '../utils/api'
import { Status } from '../constants/status'

function EventLog({ refreshKey, onRegisterPrepend }) {
  const [events, setEvents] = useState([])
  const [loading, setLoading] = useState(true)
  const [activeTab, setActiveTab] = useState('all')
  const activeTabRef = useRef(activeTab)
  activeTabRef.current = activeTab
  const eventsRef = useRef(events)
  eventsRef.current = events

  const prependEvent = useCallback((eventData) => {
    setEvents((prev) => {
      if (prev.some((e) => e._id === eventData._id || e.track_id === eventData.track_id)) {
        return prev
      }
      return [eventData, ...prev].slice(0, 100)
    })
  }, [])

  useEffect(() => {
    if (onRegisterPrepend) {
      onRegisterPrepend(prependEvent)
    }
  }, [onRegisterPrepend, prependEvent])

  useEffect(() => {
    fetchEvents()
    const interval = setInterval(fetchEvents, 60000)
    return () => clearInterval(interval)
  }, [refreshKey, activeTab])

  const fetchEvents = async () => {
    const tabAtRequestTime = activeTabRef.current
    try {
      const params = { limit: 50 }
      if (tabAtRequestTime !== 'all') {
        params.status = tabAtRequestTime
      }
      const response = await api.get('/api/events', { params })
      const dbEvents = response.data.events || []

      setEvents((prev) => {
        if (activeTabRef.current !== tabAtRequestTime) return prev
        if (prev.length === 0) return dbEvents

        const prevByTrack = new Map(prev.map((e) => [e.track_id, e]))
        const dbByTrack = new Map(dbEvents.map((e) => [e.track_id, e]))

        const merged = prev.map((e) => {
          const dbVer = dbByTrack.get(e.track_id)
          return dbVer ? { ...e, ...dbVer } : e
        })

        for (const e of dbEvents) {
          if (!prevByTrack.has(e.track_id)) merged.push(e)
        }

        return merged.slice(0, 100)
      })
    } catch (error) {
      console.error('Failed to fetch events:', error)
    } finally {
      setLoading(false)
    }
  }

  const formatTime = (dateString) => {
    if (!dateString) return 'Unknown'
    const date = new Date(dateString)
    return date.toLocaleString()
  }

  const formatTimeAgo = (dateString) => {
    if (!dateString) return ''
    const date = new Date(dateString)
    const now = new Date()
    const diffMs = now - date
    const diffSecs = Math.floor(diffMs / 1000)
    if (diffSecs < 60) return `${diffSecs}s ago`
    const diffMins = Math.floor(diffSecs / 60)
    if (diffMins < 60) return `${diffMins}m ago`
    const diffHrs = Math.floor(diffMins / 60)
    if (diffHrs < 24) return `${diffHrs}h ago`
    return `${Math.floor(diffHrs / 24)}d ago`
  }

  const getStatusClass = (status) => {
    switch (status) {
      case Status.UNKNOWN: return 'status-unverified'
      case Status.VERIFIED: return 'status-verified'
      case Status.AUTHORIZED: return 'status-authorized'
      case Status.KNOWN_VISITOR: return 'status-known_visitor'
      case Status.MASKED_UNKNOWN: return 'status-masked'
      case Status.BLACKLIST: return 'status-blacklist'
      case Status.HIDDEN: return 'status-hidden'
      default: return 'status-unverified'
    }
  }

  const getStatusLabel = (status) => {
    switch (status) {
      case Status.UNKNOWN: return 'Unverified'
      case Status.VERIFIED: return 'Verified'
      case Status.AUTHORIZED: return 'Authorized'
      case Status.KNOWN_VISITOR: return 'Known Visitor'
      case Status.MASKED_UNKNOWN: return 'Masked Unknown'
      case Status.BLACKLIST: return 'Blacklist'
      case Status.HIDDEN: return 'Hidden'
      default: return status
    }
  }

  const getStatusIcon = (status) => {
    switch (status) {
      case Status.VERIFIED: return '\u2713'
      case Status.AUTHORIZED: return '\u2713'
      case Status.KNOWN_VISITOR: return '\u2605'
      case Status.BLACKLIST: return '\u2716'
      default: return '\u26A0'
    }
  }

  const getEventItemClass = (status) => {
    switch (status) {
      case Status.BLACKLIST:
        return 'event-item-alert event-item-critical'
      case Status.UNKNOWN:
      case Status.MASKED_UNKNOWN:
      case Status.HIDDEN:
        return 'event-item-alert'
      case Status.UNCERTAIN:
        return 'event-item-uncertain'
      case Status.VERIFIED:
      case Status.AUTHORIZED:
      case Status.KNOWN_VISITOR:
        return 'event-item-verified'
      default:
        return ''
    }
  }

  return (
    <div>
      <div className="panel-header">
        <span className="panel-title">
          Activity Log
        </span>
        <span className="panel-count">{events.length} events</span>
      </div>

      <div className="tabs">
        <button
          className={`tab ${activeTab === 'all' ? 'active' : ''}`}
          onClick={() => setActiveTab('all')}
        >
          All
        </button>
        <button
          className={`tab ${activeTab === Status.UNKNOWN ? 'active' : ''}`}
          onClick={() => setActiveTab(Status.UNKNOWN)}
        >
          Unknown
        </button>
        <button
          className={`tab ${activeTab === Status.MASKED_UNKNOWN ? 'active' : ''}`}
          onClick={() => setActiveTab(Status.MASKED_UNKNOWN)}
        >
          Masked
        </button>
        <button
          className={`tab ${activeTab === Status.BLACKLIST ? 'active' : ''}`}
          onClick={() => setActiveTab(Status.BLACKLIST)}
        >
          Blacklist
        </button>
        <button
          className={`tab ${activeTab === Status.VERIFIED ? 'active' : ''}`}
          onClick={() => setActiveTab(Status.VERIFIED)}
        >
          Verified
        </button>
      </div>

      {loading ? (
        <div className="loading">Loading...</div>
      ) : events.length === 0 ? (
        <div className="empty-state">No events recorded</div>
      ) : (
        <div className="events-list">
          {events.map((event) => (
            <div key={event._id} className={`event-item ${getEventItemClass(event.status)}`}>
              {getImageUrl(event.person_crop_url || event.image_url || event.person_image) ? (
                <img
                  src={getImageUrl(event.person_crop_url || event.image_url || event.person_image)}
                  alt="Event"
                  className="event-image"
                />
              ) : (
                <div className="event-image" style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: '#666',
                  fontSize: '0.7rem'
                }}>
                  No Image
                </div>
              )}
              <div className="event-details">
                <div className="event-top-row">
                  <div className={`event-status ${getStatusClass(event.status)}`}>
                    <span className="event-status-icon">{getStatusIcon(event.status)}</span>
                    {getStatusLabel(event.status)}
                  </div>
                  <div className="event-ago">{formatTimeAgo(event.timestamp)}</div>
                </div>
                {(event.person_name || event.name) && (
                  <div className="event-person-name">
                    {event.person_name || event.name}
                  </div>
                )}
                <div className="event-full-time">
                  {formatTime(event.timestamp)}
                </div>
                {event.reason && (
                  <div className="event-reason">{event.reason}</div>
                )}
                {event.similarity_score > 0 && (
                  <div className="event-similarity">
                    Match: {Math.round(event.similarity_score * 100)}%
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

export default EventLog
