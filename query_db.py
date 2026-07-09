import sqlite3
import json

db_path = r'C:\Users\Aummc\.local\share\mimocode\mimocode.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# List tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [row[0] for row in cursor.fetchall()]
print("Tables:", tables)

# Show schema for each table
for table in tables:
    cursor.execute(f"PRAGMA table_info({table})")
    columns = cursor.fetchall()
    print(f"\nTable: {table}")
    for col in columns:
        print(f"  {col[1]} ({col[2]})")

conn.close()