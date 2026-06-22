import React, { useState, useEffect, useRef } from 'react'

function LiveFeed({ activeAlert }) {
  const [connected, setConnected] = useState(false)
  const [frame, setFrame] = useState(null)
  const [error, setError] = useState(null)
  const wsRef = useRef(null)
  const reconnectTimeoutRef = useRef(null)

  useEffect(() => {
    connectWebSocket()
    return () => {
      if (wsRef.current) {
        wsRef.current.close()
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
      }
    }
  }, [])

  const connectWebSocket = () => {
    try {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      const wsUrl = `${protocol}//${window.location.host}/ws/live`
      
      const ws = new WebSocket(wsUrl)
      wsRef.current = ws

      ws.onopen = () => {
        setConnected(true)
        setError(null)
        console.log('WebSocket connected')
        
        const pingInterval = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send('ping')
          }
        }, 30000)
        
        ws.onclose = () => clearInterval(pingInterval)
      }

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          if (data.type === 'frame' && data.data) {
            setFrame(`data:image/jpeg;base64,${data.data}`)
          }
        } catch (e) {
          console.error('Failed to parse message:', e)
        }
      }

      ws.onerror = (error) => {
        console.error('WebSocket error:', error)
        setError('Connection error')
        setConnected(false)
      }

      ws.onclose = () => {
        setConnected(false)
        reconnectTimeoutRef.current = setTimeout(connectWebSocket, 3000)
      }
    } catch (e) {
      console.error('Failed to create WebSocket:', e)
      setError('Failed to connect')
      reconnectTimeoutRef.current = setTimeout(connectWebSocket, 3000)
    }
  }

  return (
    <div>
      <div className="panel-header">
        <span className="panel-title">
          Live Camera Feed
        </span>
        <div className="header-status" style={{ fontSize: '0.85rem' }}>
          <span className="status-dot" style={{
            background: connected ? '#4ade80' : '#ef4444'
          }}></span>
          <span>{connected ? 'Connected' : 'Disconnected'}</span>
        </div>
      </div>

      <div className="feed-container">
        {frame ? (
          <img src={frame} alt="Live feed" />
        ) : (
          <div className="feed-placeholder">
            {error ? error : (connected ? 'Waiting for feed...' : 'Connecting...')}
          </div>
        )}

        {activeAlert && (
          <div className="feed-alert-overlay">
            <span className="feed-alert-icon">&#9888;</span>
            <span>UNKNOWN PERSON DETECTED - {new Date(activeAlert.timestamp).toLocaleTimeString()}</span>
          </div>
        )}
      </div>
    </div>
  )
}

export default LiveFeed
