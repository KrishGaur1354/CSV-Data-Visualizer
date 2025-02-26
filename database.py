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

# Add a comment to the database
def add_comment(name, comment):
    sentiment = analyze_sentiment(comment)
    conn = sqlite3.connect('comments.db')
    c = conn.cursor()
    c.execute("INSERT INTO comments (name, comment, sentiment) VALUES (?, ?, ?)", 
              (name, comment, sentiment))
    conn.commit()
    conn.close()

# Fetch all comments from the database
def get_comments():
    conn = sqlite3.connect('comments.db')
    c = conn.cursor()
    c.execute("SELECT id, name, comment, sentiment, timestamp FROM comments ORDER BY timestamp DESC")
    comments = c.fetchall()
    conn.close()
    return comments

# Edit a comment in the database
def edit_comment(comment_id, new_comment):
    sentiment = analyze_sentiment(new_comment)
    conn = sqlite3.connect('comments.db')
    c = conn.cursor()
    c.execute("UPDATE comments SET comment = ?, sentiment = ? WHERE id = ?", 
              (new_comment, sentiment, comment_id))
    conn.commit()
    conn.close()