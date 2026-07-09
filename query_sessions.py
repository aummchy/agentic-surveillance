import sqlite3
import json

db_path = r'C:\Users\Aummc\.local\share\mimocode\mimocode.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get recent sessions
cursor.execute("""
    SELECT id, project_id, title, time_created, time_updated
    FROM session
    ORDER BY time_created DESC
    LIMIT 10
""")
sessions = cursor.fetchall()
print("Recent sessions:")
for session in sessions:
    print(f"  ID: {session[0]}")
    print(f"  Project: {session[1]}")
    print(f"  Title: {session[2]}")
    print(f"  Created: {session[3]}")
    print(f"  Updated: {session[4]}")
    print()

# Get message count per session
cursor.execute("""
    SELECT session_id, COUNT(*) as msg_count
    FROM message
    GROUP BY session_id
    ORDER BY msg_count DESC
    LIMIT 10
""")
message_counts = cursor.fetchall()
print("Sessions with most messages:")
for mc in message_counts:
    print(f"  Session {mc[0]}: {mc[1]} messages")

conn.close()