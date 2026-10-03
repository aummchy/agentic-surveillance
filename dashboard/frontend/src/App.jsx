import React, { useState, useEffect, useRef, useCallback } from 'react'
import api, { getImageUrl } from './utils/api'
import { Status } from './constants/status'
import LiveFeed from './components/LiveFeed'
import UnknownPersons from './components/UnknownPersons'
import VerifiedPersons from './components/VerifiedPersons'
import EventLog from './components/EventLog'
import VerifyModal from './components/VerifyModal'
import ChatPanel from './components/ChatPanel'
import ErrorBoundary from './components/ErrorBoundary'

function App() {
  const [selectedPerson, setSelectedPerson] = useState(null)
  const [showModal, setShowModal] = useState(false)
  const [refreshKey, setRefreshKey] = useState(0)
  const [stats, setStats] = useState({ total_unknown: 0, total_verified: 0, events_today: 0, unknown_today: 0 })
  const [notifications, setNotifications] = useState([])
  const [activeAlert, setActiveAlert] = useState(null)
  const [liveFrame, setLiveFrame] = useState(null)
  const [wsConnected, setWsConnected] = useState(false)
  const [activeView, setActiveView] = useState('dashboard')
  const wsRef = useRef(null)
  const statsIntervalRef = useRef(null)
  const reconnectRef = useRef(null)
  const alertTimeoutRef = useRef(null)
  const liveEventPrependRef = useRef(null)
  const reconnectAttemptRef = useRef(0)
  const unmountedRef = useRef(false)
  const pendingFrameRef = useRef(null)
  const frameRafRef = useRef(null)

  const fetchStats = useCallback(async () => {
    try {
      const res = await api.get('/api/events/stats')
      if (res.status === 200) {
        setStats(res.data)
      }
    } catch (e) {
      console.error('Failed to fetch stats:', e)
    }
  }, [])

  useEffect(() => {
    fetchStats()
    statsIntervalRef.current = setInterval(fetchStats, 10000)
    return () => clearInterval(statsIntervalRef.current)
  }, [fetchStats, refreshKey])

  const connectWebSocket = useCallback(() => {
    try {
      if (unmountedRef.current) return
      if (wsRef.current) {
        try { wsRef.current.close() } catch { /* already closed */ }
        wsRef.current = null
      }
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      const ws = new WebSocket(`${protocol}//${window.location.host}/ws/live`)
      wsRef.current = ws

      ws.onopen = () => {
        reconnectAttemptRef.current = 0
        setWsConnected(true)
        if (ws._pingInterval) clearInterval(ws._pingInterval)
        const pingInterval = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) ws.send('ping')
        }, 30000)
        ws._pingInterval = pingInterval
      }

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data)
          if (msg.type === 'frame' && msg.data) {
            pendingFrameRef.current = `data:image/jpeg;base64,${msg.data}`
            if (frameRafRef.current === null) {
              frameRafRef.current = requestAnimationFrame(() => {
                frameRafRef.current = null
                const frame = pendingFrameRef.current
                if (frame) {
                  pendingFrameRef.current = null
                  setLiveFrame(frame)
                }
              })
            }
          } else if (msg.type === 'event' && msg.data) {
            if (liveEventPrependRef.current) {
              liveEventPrependRef.current(msg.data)
            }
            setRefreshKey(prev => prev + 1)
            fetchStats()
          } else if (msg.type === 'alert' && msg.data) {
            const alertData = msg.data
            setActiveAlert(alertData)
            setNotifications(prev => [alertData, ...prev].slice(0, 20))
            setRefreshKey(prev => prev + 1)
            fetchStats()
            if (alertTimeoutRef.current) clearTimeout(alertTimeoutRef.current)
            alertTimeoutRef.current = setTimeout(() => setActiveAlert(null), 8000)
          }
        } catch (e) {
          console.error('WebSocket message parse error:', e)
        }
      }

      ws.onerror = () => {
        console.warn('WebSocket connection error')
        setWsConnected(false)
      }

      ws.onclose = (e) => {
        console.warn(`WebSocket closed (code=${e.code}, reason=${e.reason || 'none'})`)
        setWsConnected(false)
        if (ws._pingInterval) clearInterval(ws._pingInterval)
        if (unmountedRef.current || wsRef.current !== ws) return
        const attempt = reconnectAttemptRef.current
        reconnectRef.current = setTimeout(
          connectWebSocket,
          Math.min(500 * 2 ** attempt, 5000)
        )
        reconnectAttemptRef.current = attempt + 1
      }
    } catch (e) {
      console.error('WebSocket init failed:', e)
      if (unmountedRef.current) return
      reconnectRef.current = setTimeout(connectWebSocket, 500)
    }
  }, [fetchStats])

  useEffect(() => {
    unmountedRef.current = false
    connectWebSocket()
    return () => {
      unmountedRef.current = true
      if (wsRef.current) {
        wsRef.current.onclose = null
        wsRef.current.close()
      }
      if (reconnectRef.current) clearTimeout(reconnectRef.current)
      if (alertTimeoutRef.current) clearTimeout(alertTimeoutRef.current)
      if (frameRafRef.current) cancelAnimationFrame(frameRafRef.current)
    }
  }, [connectWebSocket])

  const handleVerify = (person) => {
    setSelectedPerson(person)
    setShowModal(true)
  }

  const handleVerified = () => {
    setShowModal(false)
    setSelectedPerson(null)
    setRefreshKey(prev => prev + 1)
  }

  const dismissNotification = (index) => {
    setNotifications(prev => prev.filter((_, i) => i !== index))
  }

  const registerLiveEventPrepend = useCallback((prependFn) => {
    liveEventPrependRef.current = prependFn
  }, [])

  return (
    <div className="app">
      <header className="header">
        <h1>Surveillance Dashboard</h1>
        <div className="header-status">
          <span className="status-dot"></span>
          <span>System Active</span>
        </div>
      </header>

      <div className="stats-bar">
        <div className="stat-item stat-unknown">
          <div className="stat-value">{stats.total_unknown}</div>
          <div className="stat-label">Unknown Persons</div>
        </div>
        <div className="stat-item stat-verified">
          <div className="stat-value">{stats.total_verified}</div>
          <div className="stat-label">Verified Persons</div>
        </div>
        <div className="stat-item stat-events">
          <div className="stat-value">{stats.events_today}</div>
          <div className="stat-label">Events Today</div>
        </div>
        <div className="stat-item stat-alerts">
          <div className="stat-value">{stats.unknown_today}</div>
          <div className="stat-label">Unknowns Today</div>
        </div>
      </div>

      <div className="nav-tabs">
        <button
          className={`nav-tab ${activeView === 'dashboard' ? 'active' : ''}`}
          onClick={() => setActiveView('dashboard')}
        >
          Dashboard
        </button>
        <button
          className={`nav-tab ${activeView === 'manage' ? 'active' : ''}`}
          onClick={() => setActiveView('manage')}
        >
          Manage
        </button>
      </div>

      {activeAlert && (
        <div className="notification-popup" key={activeAlert.timestamp}>
          <div className="notification-header">
            <span className="notification-icon">&#9888;</span>
            <span className="notification-title">
              {activeAlert.alert_level === 'critical' ? 'CRITICAL Alert' :
               activeAlert.alert_level === 'high' ? 'High Alert' :
               activeAlert.status === Status.MASKED_UNKNOWN ? 'Masked Unknown Detected' : 'Unknown Person Detected'}
            </span>
            <button className="notification-close" onClick={() => setActiveAlert(null)}>&times;</button>
          </div>
          <div className="notification-body">
            {(activeAlert.person_crop_url || activeAlert.image_url) && (
              <img src={getImageUrl(activeAlert.person_crop_url || activeAlert.image_url)} alt="Unknown" className="notification-image" />
            )}
            <div className="notification-details">
              <div className="notification-time">
                {new Date(activeAlert.timestamp).toLocaleTimeString()}
              </div>
              <div className="notification-camera">{activeAlert.camera_id}</div>
              {activeAlert.nl_summary ? (
                <div className="notification-nl-summary">{activeAlert.nl_summary}</div>
              ) : (
                <div className="notification-reason">{activeAlert.reason}</div>
              )}
            </div>
          </div>
        </div>
      )}

      <main className="main-content">
        {activeView === 'dashboard' && (
          <>
            <section className="panel live-feed">
              <ErrorBoundary label="Live Feed">
                <LiveFeed frame={liveFrame} connected={wsConnected} activeAlert={activeAlert} />
              </ErrorBoundary>
            </section>

            <section className="panel activity-log">
              <ErrorBoundary label="Activity Log">
                <EventLog refreshKey={refreshKey} onRegisterPrepend={registerLiveEventPrepend} />
              </ErrorBoundary>
            </section>

            <section className="panel chat-section">
              <ErrorBoundary label="Chat Panel">
                <ChatPanel />
              </ErrorBoundary>
            </section>
          </>
        )}

        {activeView === 'manage' && (
          <div className="three-col-row">
            <section className="panel">
              <ErrorBoundary label="Unknown Persons">
                <UnknownPersons
                  onVerify={handleVerify}
                  refreshKey={refreshKey}
                />
              </ErrorBoundary>
            </section>

            <section className="panel">
              <ErrorBoundary label="Verified Persons">
                <VerifiedPersons refreshKey={refreshKey} />
              </ErrorBoundary>
            </section>

            <section className="panel">
              <ErrorBoundary label="Event Log">
                <EventLog refreshKey={refreshKey} onRegisterPrepend={registerLiveEventPrepend} />
              </ErrorBoundary>
            </section>
          </div>
        )}
      </main>

      {notifications.length > 0 && (
        <div className="notification-sidebar">
          <div className="notification-sidebar-header">
            <span>Alerts</span>
            <button className="clear-btn" onClick={() => setNotifications([])}>Clear All</button>
          </div>
          {notifications.map((n, i) => (
            <div key={n.timestamp + i} className="notification-sidebar-item" onClick={() => dismissNotification(i)}>
              <div className="notif-img-wrap">
                {(n.person_crop_url || n.image_url) && <img src={getImageUrl(n.person_crop_url || n.image_url)} alt="" className="notif-thumb" />}
              </div>
              <div className="notif-info">
                <div className="notif-label">
                  {n.status === Status.MASKED_UNKNOWN ? 'Masked Unknown' :
                   n.alert_level === 'critical' ? 'CRITICAL' :
                   n.alert_level === 'high' ? 'High Alert' : 'Unknown'}
                </div>
                {n.nl_summary ? (
                  <div className="notif-nl-summary">{n.nl_summary}</div>
                ) : (
                  <div className="notif-time">{new Date(n.timestamp).toLocaleTimeString()}</div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}

      {showModal && (
        <VerifyModal
          person={selectedPerson}
          onVerified={handleVerified}
          onClose={() => {
            setShowModal(false)
            setSelectedPerson(null)
          }}
        />
      )}
    </div>
  )
}

export default App
