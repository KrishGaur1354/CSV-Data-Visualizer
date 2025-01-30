import sqlite3
from textblob import TextBlob
from datetime import datetime

# Initialize the database
def init_db():
    conn = sqlite3.connect('comments.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS comments
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  name TEXT, 
                  comment TEXT, 
                  sentiment TEXT, 
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

# Analyze sentiment using TextBlob
def analyze_sentiment(comment):
    analysis = TextBlob(comment)
    if analysis.sentiment.polarity > 0:
        return "positive"
    elif analysis.sentiment.polarity < 0:
        return "negative"
    else:
        return "neutral"

def add_comment(name, comment):
    try:
        sentiment = analyze_sentiment(comment)
        conn = sqlite3.connect('comments.db')
        c = conn.cursor()
        c.execute("INSERT INTO comments (name, comment, sentiment) VALUES (?, ?, ?)", 
                  (name, comment, sentiment))
        conn.commit()
    except Exception as e:
        print(f"Error adding comment: {e}")
    finally:
        conn.close()

def get_comments():
    try:
        conn = sqlite3.connect('comments.db')
        c = conn.cursor()
        c.execute("SELECT name, comment, sentiment, timestamp FROM comments ORDER BY timestamp DESC")
        comments = c.fetchall()
        return comments
    except Exception as e:
        print(f"Error fetching comments: {e}")
        return []
    finally:
        conn.close()