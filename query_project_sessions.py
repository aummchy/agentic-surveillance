import sqlite3
import json

db_path = r'C:\Users\Aummc\.local\share\mimocode\mimocode.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

project_id = '79cf516b-d95d-4ffa-9994-ee47a9aeac95'

# Get all sessions for this project
cursor.execute("""
    SELECT id, title, time_created, time_updated
    FROM session
    WHERE project_id = ?
    ORDER BY time_created DESC
""", (project_id,))
sessions = cursor.fetchall()

print(f"Sessions for project {project_id}:")
for session in sessions:
    print(f"  ID: {session[0]}")
    print(f"  Title: {session[1]}")
    print(f"  Created: {session[2]}")
    print(f"  Updated: {session[3]}")
    print()

conn.close()