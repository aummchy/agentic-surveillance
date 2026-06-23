import React, { useState, useEffect, useRef, useCallback } from 'react'
import LiveFeed from './components/LiveFeed'
import UnknownPersons from './components/UnknownPersons'
import EventLog from './components/EventLog'
import VerifyModal from './components/VerifyModal'

function App() {
  const [selectedPerson, setSelectedPerson] = useState(null)
  const [showModal, setShowModal] = useState(false)
  const [unknowns, setUnknowns] = useState([])
  const [refreshKey, setRefreshKey] = useState(0)
  const [stats, setStats] = useState({ total_unknown: 0, total_verified: 0, events_today: 0, unknown_today: 0 })
  const [notifications, setNotifications] = useState([])
  const [activeAlert, setActiveAlert] = useState(null)
  const [liveFrame, setLiveFrame] = useState(null)
  const [wsConnected, setWsConnected] = useState(false)
  const wsRef = useRef(null)
  const statsIntervalRef = useRef(null)
  const reconnectRef = useRef(null)

  const fetchStats = useCallback(async () => {
    try {
      const res = await fetch('/api/events/stats')
      const data = await res.json()
      setStats(data)
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
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      const ws = new WebSocket(`${protocol}//${window.location.host}/ws/live`)
      wsRef.current = ws

      ws.onopen = () => {
        setWsConnected(true)
        const pingInterval = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) ws.send('ping')
        }, 30000)
        ws._pingInterval = pingInterval
      }

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data)
          if (msg.type === 'frame' && msg.data) {
            setLiveFrame(`data:image/jpeg;base64,${msg.data}`)
          } else if (msg.type === 'alert' && msg.data) {
            const alertData = msg.data
            setActiveAlert(alertData)
            setNotifications(prev => [alertData, ...prev].slice(0, 20))
            setRefreshKey(prev => prev + 1)
            fetchStats()
            setTimeout(() => setActiveAlert(null), 8000)
          }
        } catch (e) {}
      }

      ws.onerror = () => setWsConnected(false)

      ws.onclose = () => {
        setWsConnected(false)
        clearInterval(ws._pingInterval)
        reconnectRef.current = setTimeout(connectWebSocket, 3000)
      }
    } catch (e) {
      reconnectRef.current = setTimeout(connectWebSocket, 3000)
    }
  }, [fetchStats])

  useEffect(() => {
    connectWebSocket()
    return () => {
      if (wsRef.current) wsRef.current.close()
      if (reconnectRef.current) clearTimeout(reconnectRef.current)
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

      {activeAlert && (
        <div className="notification-popup" key={activeAlert.timestamp}>
          <div className="notification-header">
            <span className="notification-icon">&#9888;</span>
            <span className="notification-title">Unknown Person Detected</span>
            <button className="notification-close" onClick={() => setActiveAlert(null)}>&times;</button>
          </div>
          <div className="notification-body">
            {activeAlert.image_url && (
              <img src={activeAlert.image_url} alt="Unknown" className="notification-image" />
            )}
            <div className="notification-details">
              <div className="notification-time">
                {new Date(activeAlert.timestamp).toLocaleTimeString()}
              </div>
              <div className="notification-camera">{activeAlert.camera_id}</div>
              <div className="notification-reason">{activeAlert.reason}</div>
            </div>
          </div>
        </div>
      )}

      <main className="main-content">
        <section className="panel live-feed">
          <LiveFeed frame={liveFrame} connected={wsConnected} activeAlert={activeAlert} />
        </section>

        <section className="panel">
          <UnknownPersons
            onVerify={handleVerify}
            refreshKey={refreshKey}
            onUnknownsLoaded={setUnknowns}
          />
        </section>

        <section className="panel">
          <EventLog refreshKey={refreshKey} />
        </section>
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
                {n.image_url && <img src={n.image_url} alt="" className="notif-thumb" />}
              </div>
              <div className="notif-info">
                <div className="notif-label">{n.status === 'masked_unknown' ? 'Masked Unknown' : 'Unknown'}</div>
                <div className="notif-time">{new Date(n.timestamp).toLocaleTimeString()}</div>
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
