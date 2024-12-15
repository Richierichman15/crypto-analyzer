from textblob import TextBlob

class SentimentAnalyzer:
    def analyze_text(self, text):
        """Analyze sentiment of text using TextBlob"""
        blob = TextBlob(text)
        return {
            'polarity': blob.sentiment.polarity,  # -1 to 1
            'subjectivity': blob.sentiment.subjectivity  # 0 to 1
        }

    def get_sentiment_score(self, sentiment_data):
        """Convert sentiment analysis into a score"""
        total_polarity = 0
        total_weight = 0
        
        for data in sentiment_data:
            sentiment = self.analyze_text(data['content'])
            # Give more weight to more objective content
            weight = 1 - sentiment['subjectivity']
            total_polarity += sentiment['polarity'] * weight
            total_weight += weight
            
        return total_polarity / total_weight if total_weight > 0 else 0