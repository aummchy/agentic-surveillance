import sqlite3
import json

db_path = r'C:\Users\Aummc\.local\share\mimocode\mimocode.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

session_id = 'ses_0ba8c99a9ffegTICkG5GPrW54k'

# Get session info
cursor.execute("""
    SELECT id, project_id, title, time_created, time_updated
    FROM session
    WHERE id = ?
""", (session_id,))
session = cursor.fetchone()
if session:
    print(f"Session: {session[0]}")
    print(f"Project: {session[1]}")
    print(f"Title: {session[2]}")
    print(f"Created: {session[3]}")
    print(f"Updated: {session[4]}")
    print()

# Get messages in this session
cursor.execute("""
    SELECT id, agent_id, time_created, data
    FROM message
    WHERE session_id = ?
    ORDER BY time_created
""", (session_id,))
messages = cursor.fetchall()

print(f"Messages in session {session_id}:")
for msg in messages[:5]:  # Show first 5 messages
    msg_id, agent_id, time_created, data = msg
    data_dict = json.loads(data) if data else {}
    role = data_dict.get('role', 'unknown')
    print(f"  ID: {msg_id}")
    print(f"  Agent: {agent_id or 'main'}")
    print(f"  Role: {role}")
    print(f"  Created: {time_created}")
    print()

conn.close()