from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import numpy as np
import joblib
from typing import List, Dict, Tuple
import sys
import os

# Add the parent directory to the path to import from other modules
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from model.feature_extractor import EmailFeatureExtractor


class EmailMultiClassifier:
    def __init__(self):
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        self.priority_encoder = LabelEncoder()
        self.mlb = MultiLabelBinarizer()  # For multi-label tag encoding
        self.feature_extractor = EmailFeatureExtractor()
        self.feature_names = []
        self.is_trained = False

    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare feature matrix from email DataFrame"""
        features_list = []

        print(f"📊 Preparing features for {len(df)} emails...")

        for _, email in df.iterrows():
            email_dict = {
                'subject': email['subject'],
                'body': email['body'],
                'from': email['from']
            }

            # Extract features using our feature extractor
            email_features = self.feature_extractor.combine_features(email_dict)
            features_list.append(email_features)

        # Convert to DataFrame to handle missing features
        features_df = pd.DataFrame(features_list)

        # Fill missing values with 0
        features_df = features_df.fillna(0)

        # Store feature names for later use
        self.feature_names = list(features_df.columns)

        print(f"✅ Created feature matrix with {features_df.shape[1]} features")

        return features_df.values

    def train(self, df: pd.DataFrame) -> Dict:
        """Train the multi-label classifier"""
        print("🎯 Training Email Multi-Classifier")
        print("=" * 40)

        if df.empty:
            print("❌ No training data provided!")
            return {}

        # Prepare features
        X = self.prepare_features(df)

        # Prepare priority labels
        y_priority = self.priority_encoder.fit_transform(df['priority'])

        # Prepare tag labels (multi-label)
        tag_lists = df['tags'].tolist()
        y_tags = self.mlb.fit_transform(tag_lists)

        print(f"📊 Training data shape: {X.shape}")
        print(f"🎯 Priority classes: {self.priority_encoder.classes_}")
        print(f"🏷️  Tag classes: {self.mlb.classes_}")

        # Split data for validation
        if len(df) > 3:  # Only split if we have enough data
            X_train, X_val, y_priority_train, y_priority_val, y_tags_train, y_tags_val = train_test_split(
                X, y_priority, y_tags, test_size=0.2, random_state=42
            )
        else:
            # Use all data for training if dataset is small
            X_train, X_val = X, X
            y_priority_train, y_priority_val = y_priority, y_priority
            y_tags_train, y_tags_val = y_tags, y_tags

        # Train priority classifier
        print("🔄 Training priority classifier...")
        self.priority_classifier = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        self.priority_classifier.fit(X_train, y_priority_train)

        # Train tag classifier
        print("🔄 Training tag classifier...")
        self.tag_classifier = MultiOutputClassifier(
            RandomForestClassifier(n_estimators=50, random_state=42, max_depth=8)
        )
        self.tag_classifier.fit(X_train, y_tags_train)

        # Evaluate on validation set
        y_priority_pred = self.priority_classifier.predict(X_val)
        y_tags_pred = self.tag_classifier.predict(X_val)

        priority_accuracy = accuracy_score(y_priority_val, y_priority_pred)

        print(f"✅ Priority classification accuracy: {priority_accuracy:.2%}")

        self.is_trained = True

        return {
            'priority_accuracy': priority_accuracy,
            'priority_classes': self.priority_encoder.classes_.tolist(),
            'tag_classes': self.mlb.classes_.tolist(),
            'feature_count': len(self.feature_names)
        }

    def predict(self, emails: List[Dict]) -> List[Dict]:
        """Predict priority and tags for new emails"""
        if not self.is_trained:
            print("❌ Model not trained! Please train the model first.")
            return []

        print(f"🔮 Predicting for {len(emails)} emails...")

        # Convert to DataFrame format expected by prepare_features
        df = pd.DataFrame(emails)

        # Prepare features
        X = self.prepare_features(df)

        # Predict priorities
        priority_predictions = self.priority_classifier.predict(X)
        priority_labels = self.priority_encoder.inverse_transform(priority_predictions)

        # Predict tags
        tag_predictions = self.tag_classifier.predict(X)
        tag_labels = self.mlb.inverse_transform(tag_predictions)

        # Combine predictions with original emails
        results = []
        for i, email in enumerate(emails):
            result = email.copy()
            result['predicted_priority'] = priority_labels[i]
            result['predicted_tags'] = list(tag_labels[i])
            results.append(result)

        return results

    def save_model(self, path: str):
        """Save trained model"""
        if not self.is_trained:
            print("❌ Cannot save untrained model!")
            return

        model_data = {
            'priority_classifier': self.priority_classifier,
            'tag_classifier': self.tag_classifier,
            'priority_encoder': self.priority_encoder,
            'mlb': self.mlb,
            'feature_names': self.feature_names,
            'feature_extractor': self.feature_extractor
        }

        joblib.dump(model_data, path)
        print(f"✅ Model saved to {path}")

    def load_model(self, path: str):
        """Load trained model"""
        try:
            model_data = joblib.load(path)
            self.priority_classifier = model_data['priority_classifier']
            self.tag_classifier = model_data['tag_classifier']
            self.priority_encoder = model_data['priority_encoder']
            self.mlb = model_data['mlb']
            self.feature_names = model_data['feature_names']
            self.feature_extractor = model_data['feature_extractor']
            self.is_trained = True
            print(f"✅ Model loaded from {path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")


# Test the classifier
if __name__ == "__main__":
    print("🤖 Testing Email Multi-Classifier")
    print("=" * 50)

    # Import sample data
    from app.email_ingest import EmailIngestor

    # Load sample emails
    ingestor = EmailIngestor()
    emails = ingestor.load_sample_emails()

    if emails:
        df = ingestor.convert_to_dataframe(emails)

        # Initialize and train classifier
        classifier = EmailMultiClassifier()
        training_results = classifier.train(df)

        print("\n🎯 Training Results:")
        for key, value in training_results.items():
            print(f"  {key}: {value}")

        # Test prediction on first email
        test_email = [{
            'subject': 'Test: New project proposal',
            'body': 'We have a new client project that needs immediate attention. Please review the proposal.',
            'from': 'manager@company.com'
        }]

        predictions = classifier.predict(test_email)
        if predictions:
            pred = predictions[0]
            print(f"\n🔮 Test Prediction:")
            print(f"  Subject: {pred['subject']}")
            print(f"  Predicted Priority: {pred['predicted_priority']}")
            print(f"  Predicted Tags: {pred['predicted_tags']}")

        print("\n✅ Multi-classifier system working!")
    else:
        print("❌ No sample data found!")
