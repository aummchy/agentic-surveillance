import React, { useState, useRef, useEffect } from 'react'
import api from '../utils/api'

const MAX_MESSAGES = 100

function ChatPanel() {
  const [messages, setMessages] = useState([
    { role: 'assistant', text: 'Surveillance assistant ready. Ask me about events, persons, or system status.' }
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [llmAvailable, setLlmAvailable] = useState(true)
  const messagesEndRef = useRef(null)
  const inputRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const checkHealth = async () => {
    try {
      const res = await api.get('/api/chat/health')
      setLlmAvailable(res.data.available)
    } catch {
      setLlmAvailable(false)
    }
  }

  useEffect(() => {
    checkHealth()
    const interval = setInterval(checkHealth, 30000)
    return () => clearInterval(interval)
  }, [])

  const sendMessage = async () => {
    const text = input.trim()
    if (!text || loading) return

    setInput('')
    const userMsg = { role: 'user', text }
    setMessages(prev => {
      const next = [...prev, userMsg]
      return next.length > MAX_MESSAGES ? next.slice(-MAX_MESSAGES) : next
    })
    setLoading(true)

    try {
      const history = messages
        .filter((m) => m.role === 'user' || m.role === 'assistant')
        .slice(-20)
        .map((m) => ({ role: m.role, content: m.text }))

      const res = await api.post('/api/chat', { message: text, history })
      const { response, data, llm_available } = res.data
      setLlmAvailable(llm_available)
      setMessages(prev => {
        const next = [...prev, { role: 'assistant', text: response, data }]
        return next.length > MAX_MESSAGES ? next.slice(-MAX_MESSAGES) : next
      })
    } catch (err) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        text: 'Failed to get response. Is Ollama running?'
      }])
    } finally {
      setLoading(false)
      inputRef.current?.focus()
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  const suggestedQueries = [
    'How many unknowns today?',
    'Show recent events',
    'System status',
    'Recent unknown persons'
  ]

  const handleSuggestion = (q) => {
    setInput(q)
    inputRef.current?.focus()
  }

  return (
    <div className="chat-panel">
      <div className="chat-header">
        <span className="chat-title">
          <span className="chat-icon">&#128172;</span>
          AI Assistant
        </span>
        <span className={`chat-status ${llmAvailable ? 'chat-online' : 'chat-offline'}`}>
          {llmAvailable ? 'Online' : 'Offline'}
        </span>
      </div>

      <div className="chat-messages">
        {messages.map((msg, i) => (
          <div key={i} className={`chat-msg chat-msg-${msg.role}`}>
            <div className="chat-msg-bubble">
              {msg.text}
            </div>
            {msg.data && (
              <div className="chat-data-badge">
                {msg.data.type === 'stats' && 'Data: Dashboard stats'}
                {msg.data.type === 'events' && `Data: ${msg.data.events?.length || 0} events`}
                {msg.data.type === 'unknown_faces' && `Data: ${msg.data.total || 0} unknown persons`}
                {msg.data.type === 'memory_stats' && 'Data: Memory stats'}
              </div>
            )}
          </div>
        ))}
        {loading && (
          <div className="chat-msg chat-msg-assistant">
            <div className="chat-msg-bubble chat-typing">
              <span className="dot"></span>
              <span className="dot"></span>
              <span className="dot"></span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {messages.length <= 1 && (
        <div className="chat-suggestions">
          {suggestedQueries.map((q, i) => (
            <button key={i} className="chat-suggestion" onClick={() => handleSuggestion(q)}>
              {q}
            </button>
          ))}
        </div>
      )}

      <div className="chat-input-area">
        <input
          ref={inputRef}
          type="text"
          className="chat-input"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={llmAvailable ? 'Ask about events, persons, status...' : 'LLM offline'}
          disabled={!llmAvailable}
        />
        <button
          className="chat-send"
          onClick={sendMessage}
          disabled={loading || !input.trim() || !llmAvailable}
        >
          &#10148;
        </button>
      </div>
    </div>
  )
}

export default ChatPanel
