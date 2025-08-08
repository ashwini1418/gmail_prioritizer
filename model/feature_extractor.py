import nltk
import spacy
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, List
import numpy as np

# Download required NLTK data (run once)
# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)


class EmailFeatureExtractor:
    def __init__(self):
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
            print("✅ spaCy model loaded successfully")
        except OSError:
            print("❌ Please install spaCy English model: python -m spacy download en_core_web_sm")
            self.nlp = None

        self.tfidf = TfidfVectorizer(max_features=1000, stop_words='english')

    def extract_text_features(self, text: str) -> Dict:
        """Extract comprehensive NLP features from text"""
        features = {}

        # Basic text statistics
        features['char_count'] = len(text)
        features['word_count'] = len(text.split())
        features['sentence_count'] = len(nltk.sent_tokenize(text))
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0

        # Urgency and priority indicators
        urgent_words = ['urgent', 'asap', 'immediately', 'emergency', 'critical', 'deadline', 'rush']
        features['urgency_score'] = sum(1 for word in urgent_words if word in text.lower())

        # Business/work indicators
        work_words = ['meeting', 'project', 'deadline', 'report', 'client', 'presentation', 'budget']
        features['work_score'] = sum(1 for word in work_words if word in text.lower())

        # Personal/social indicators
        personal_words = ['birthday', 'congratulations', 'family', 'friend', 'vacation', 'weekend']
        features['personal_score'] = sum(1 for word in personal_words if word in text.lower())

        # Promotional/spam indicators
        promo_words = ['sale', 'discount', 'offer', 'free', 'limited time', 'click here', 'buy now']
        features['promo_score'] = sum(1 for word in promo_words if word in text.lower())

        # Question indicators
        features['question_marks'] = text.count('?')
        features['has_questions'] = '?' in text

        # Sentiment analysis
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            sia = SentimentIntensityAnalyzer()
            sentiment = sia.polarity_scores(text)
            features.update({
                'sentiment_positive': sentiment['pos'],
                'sentiment_negative': sentiment['neg'],
                'sentiment_neutral': sentiment['neu'],
                'sentiment_compound': sentiment['compound']
            })
        except:
            features.update({
                'sentiment_positive': 0,
                'sentiment_negative': 0,
                'sentiment_neutral': 1,
                'sentiment_compound': 0
            })

        return features

    def extract_sender_features(self, sender_email: str) -> Dict:
        """Extract features from sender email"""
        domain = sender_email.split('@')[1] if '@' in sender_email else 'unknown'

        # Domain categorization - Convert to numerical features
        domain_categories = {
            'gmail.com': 'personal',
            'yahoo.com': 'personal',
            'hotmail.com': 'personal',
            'outlook.com': 'personal',
            'company.com': 'work',
            'work.com': 'work',
        }

        sender_type = domain_categories.get(domain, 'unknown')

        # Convert categorical features to numerical (0/1 binary features)
        features = {
            'sender_is_personal': 1 if sender_type == 'personal' else 0,
            'sender_is_work': 1 if sender_type == 'work' else 0,
            'sender_is_unknown': 1 if sender_type == 'unknown' else 0,
            'is_noreply': 1 if 'noreply' in sender_email.lower() else 0,
            'is_support': 1 if any(word in sender_email.lower() for word in ['support', 'help', 'service']) else 0,
            'is_sales': 1 if any(word in sender_email.lower() for word in ['sales', 'marketing', 'promo']) else 0,
            'is_security': 1 if any(word in sender_email.lower() for word in ['security', 'alert', 'warning']) else 0,
            'domain_length': len(domain)  # Numerical feature for domain length
        }

        return features

    def extract_subject_features(self, subject: str) -> Dict:
        """Extract specific features from email subject"""
        features = {}

        # Subject-specific patterns
        features['subject_has_urgent'] = any(word in subject.lower() for word in ['urgent', 'important', 'asap'])
        features['subject_has_re'] = subject.lower().startswith('re:')
        features['subject_has_fwd'] = subject.lower().startswith('fwd:') or 'forward' in subject.lower()
        features['subject_all_caps'] = subject.isupper()
        features['subject_has_numbers'] = bool(re.search(r'\d', subject))
        features['subject_exclamation_marks'] = subject.count('!')

        return features

    def combine_features(self, email: Dict) -> Dict:
        """Combine all features for an email"""
        # Extract combined text
        combined_text = f"{email['subject']} {email['body']}"

        # Get all feature sets
        text_features = self.extract_text_features(combined_text)
        sender_features = self.extract_sender_features(email['from'])
        subject_features = self.extract_subject_features(email['subject'])

        # Combine all features
        all_features = {**text_features, **sender_features, **subject_features}

        return all_features


# Test the feature extractor
if __name__ == "__main__":
    print("🧪 Testing Email Feature Extractor")
    print("=" * 50)

    # Sample email for testing
    sample_email = {
        'subject': 'Urgent: Project deadline approaching!',
        'body': 'Hi team, we need to finish the quarterly report by tomorrow. Please send me your sections ASAP.',
        'from': 'boss@company.com'
    }

    # Initialize feature extractor
    extractor = EmailFeatureExtractor()

    # Extract features
    features = extractor.combine_features(sample_email)

    print(f"✅ Extracted {len(features)} features:")
    for key, value in list(features.items())[:10]:  # Show first 10 features
        print(f"  {key}: {value}")

    print("\n🎯 Key insights:")
    print(f"  Urgency Score: {features.get('urgency_score', 0)}")
    print(f"  Work Score: {features.get('work_score', 0)}")
    print(f"  Sentiment: {features.get('sentiment_compound', 0):.2f}")
    print(f"  Sender Type: {features.get('sender_type', 'unknown')}")
