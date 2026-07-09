import sqlite3
import json

db_path = r'C:\Users\Aummc\.local\share\mimocode\mimocode.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

session_id = 'ses_0ba8c9d0fffeyW7OAO5Emq7HYC'

# Get parts with text content
cursor.execute("""
    SELECT p.id, p.message_id, p.time_created, p.data
    FROM part p
    JOIN message m ON p.message_id = m.id
    WHERE m.session_id = ?
      AND json_extract(p.data, '$.type') = 'text'
    ORDER BY m.time_created, p.time_created
""", (session_id,))
parts = cursor.fetchall()

print(f"Text parts in session {session_id}:")
for part in parts:
    part_id, msg_id, time_created, data = part
    data_dict = json.loads(data) if data else {}
    text = data_dict.get('text', '')
    print(f"  Part ID: {part_id}")
    print(f"  Message ID: {msg_id}")
    print(f"  Created: {time_created}")
    print(f"  Text: {text[:300]}...")
    print()

conn.close()