import sqlite3
import json

db_path = r'C:\Users\Aummc\.local\share\mimocode\mimocode.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

session_id = 'ses_0ba85eecbffepykjWmfG0dUDrd'

# Get parts for the first few messages
cursor.execute("""
    SELECT p.id, p.message_id, p.time_created, p.data
    FROM part p
    JOIN message m ON p.message_id = m.id
    WHERE m.session_id = ?
    ORDER BY m.time_created, p.time_created
    LIMIT 5
""", (session_id,))
parts = cursor.fetchall()

print(f"Parts in session {session_id}:")
for part in parts:
    part_id, msg_id, time_created, data = part
    data_dict = json.loads(data) if data else {}
    part_type = data_dict.get('type', 'unknown')
    print(f"  Part ID: {part_id}")
    print(f"  Message ID: {msg_id}")
    print(f"  Created: {time_created}")
    print(f"  Type: {part_type}")
    
    # Show preview of content
    if part_type == 'text':
        text = data_dict.get('text', '')
        print(f"  Text preview: {text[:100]}...")
    elif part_type == 'tool':
        tool_name = data_dict.get('tool', 'unknown')
        print(f"  Tool: {tool_name}")
    print()

conn.close()