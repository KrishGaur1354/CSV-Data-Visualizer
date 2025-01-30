import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import threading

# Download required NLTK data in a separate thread to avoid blocking
def download_nltk_data():
    nltk.download('vader_lexicon')

# Start download in background
threading.Thread(target=download_nltk_data).start()

def analyze_sentiment(text):
    """
    Analyze the sentiment of given text and return sentiment scores and appropriate response
    """
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)
    compound_score = sentiment_scores['compound']
    
    if compound_score >= 0.05:
        response = {
            'sentiment': 'positive',
            'message': "Thank you for your positive feedback! We're glad you had a great experience.",
            'color': '#28a745'  # Green for positive
        }
    elif compound_score <= -0.05:
        response = {
            'sentiment': 'negative',
            'message': "We apologize for your experience. We'll work on improving our services.",
            'color': '#dc3545'  # Red for negative
        }
    else:
        response = {
            'sentiment': 'neutral',
            'message': "Thank you for your feedback. We appreciate your input.",
            'color': '#6c757d'  # Grey for neutral
        }
    
    response['score'] = compound_score
    return response