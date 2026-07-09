import sqlite3
import json

db_path = r'C:\Users\Aummc\.local\share\mimocode\mimocode.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

session_id = 'ses_0ba85eecbffepykjWmfG0dUDrd'

# Get messages in current session
cursor.execute("""
    SELECT id, agent_id, time_created, data
    FROM message
    WHERE session_id = ?
    ORDER BY time_created
""", (session_id,))
messages = cursor.fetchall()

print(f"Messages in session {session_id}:")
for msg in messages:
    msg_id, agent_id, time_created, data = msg
    data_dict = json.loads(data) if data else {}
    role = data_dict.get('role', 'unknown')
    print(f"  ID: {msg_id}")
    print(f"  Agent: {agent_id or 'main'}")
    print(f"  Role: {role}")
    print(f"  Created: {time_created}")
    print()

conn.close()