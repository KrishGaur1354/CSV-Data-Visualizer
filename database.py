# database.py
import sqlite3

# Initialize the database
def init_db():
    conn = sqlite3.connect('comments.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS comments
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, comment TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

# Add a comment to the database
def add_comment(name, comment):
    conn = sqlite3.connect('comments.db')
    c = conn.cursor()
    c.execute("INSERT INTO comments (name, comment) VALUES (?, ?)", (name, comment))
    conn.commit()
    conn.close()

# Fetch all comments from the database
def get_comments():
    conn = sqlite3.connect('comments.db')
    c = conn.cursor()
    c.execute("SELECT name, comment, timestamp FROM comments ORDER BY timestamp DESC")
    comments = c.fetchall()
    conn.close()
    return comments