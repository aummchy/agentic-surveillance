import React from 'react'

function LiveFeed({ frame, connected, activeAlert }) {
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
            {connected ? 'Waiting for feed...' : 'Connecting...'}
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
