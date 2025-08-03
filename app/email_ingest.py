import json
import pandas as pd
from datetime import datetime
from typing import List, Dict
import os


class EmailIngestor:
    def __init__(self, data_path: str = 'data/sample_emails.json'):
        self.data_path = data_path
        # Also try absolute path as backup
        if not os.path.exists(data_path):
            # Try from project root
            project_root = os.path.dirname(os.path.dirname(__file__))  # Go up two levels from app/
            self.data_path = os.path.join(project_root, data_path)
            print(f"🔍 Trying alternate path: {self.data_path}")

    def load_sample_emails(self) -> List[Dict]:
        """Load emails from JSON file"""
        print(f"🔍 Looking for file at: {self.data_path}")
        print(f"📁 File exists: {os.path.exists(self.data_path)}")
        print(f"📂 Current working directory: {os.getcwd()}")

        try:
            with open(self.data_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                print(f"✅ Successfully loaded {len(data)} emails from JSON")
                return data
        except FileNotFoundError:
            print(f"❌ Error: Could not find {self.data_path}")
            print("📋 Let's list what files exist in the data directory:")
            try:
                data_dir = os.path.dirname(self.data_path)
                if os.path.exists(data_dir):
                    files = os.listdir(data_dir)
                    print(f"📁 Files in {data_dir}: {files}")
                else:
                    print(f"❌ Data directory {data_dir} doesn't exist")
            except:
                pass
            return []
        except json.JSONDecodeError as e:
            print(f"❌ Error: Invalid JSON in {self.data_path}: {e}")
            return []

    def preprocess_email(self, email: Dict) -> Dict:
        """Clean and preprocess email data"""
        # Extract basic features
        email['subject_length'] = len(email['subject'])
        email['body_length'] = len(email['body'])
        email['word_count'] = len(email['body'].split())

        # Extract domain from sender
        email['sender_domain'] = email['from'].split('@')[1] if '@' in email['from'] else 'unknown'

        # Check for urgency keywords
        urgent_keywords = ['urgent', 'asap', 'emergency', 'critical', 'immediate']
        email['has_urgent_keywords'] = any(keyword in email['subject'].lower() or
                                           keyword in email['body'].lower()
                                           for keyword in urgent_keywords)

        # Convert date string to datetime if needed
        if isinstance(email.get('date'), str):
            try:
                email['date_parsed'] = datetime.fromisoformat(email['date'].replace('Z', '+00:00'))
            except:
                email['date_parsed'] = datetime.now()

        return email

    def convert_to_dataframe(self, emails: List[Dict]) -> pd.DataFrame:
        """Convert emails to pandas DataFrame for ML processing"""
        if not emails:
            return pd.DataFrame()

        processed_emails = [self.preprocess_email(email.copy()) for email in emails]
        df = pd.DataFrame(processed_emails)

        # Display basic statistics
        if not df.empty:
            print(f"📧 Loaded {len(df)} emails")
            print(f"📊 Priority distribution:")
            print(df['priority'].value_counts())
            print(f"📋 Most common tags:")
            all_tags = [tag for tags in df['tags'] for tag in tags]
            tag_counts = pd.Series(all_tags).value_counts()
            print(tag_counts.head())

        return df

    def display_emails(self, emails: List[Dict], limit: int = 3):
        """Display emails in a readable format"""
        print(f"\n📬 Displaying {min(limit, len(emails))} sample emails:\n")

        for i, email in enumerate(emails[:limit]):
            print(f"{'=' * 50}")
            print(f"📧 Email {email['id']}")
            print(f"📋 Subject: {email['subject']}")
            print(f"👤 From: {email['from']}")
            print(f"⏰ Date: {email['date']}")
            print(f"🎯 Priority: {email['priority']}")
            print(f"🏷️  Tags: {', '.join(email['tags'])}")
            print(f"📝 Body: {email['body'][:100]}...")
            print()


# Main execution and testing
if __name__ == "__main__":
    # Test the email ingestion system
    print("🚀 Testing Email Prioritizer - Data Ingestion System")
    print("=" * 60)

    # Initialize the ingestion system
    ingestor = EmailIngestor()

    # Load sample emails
    emails = ingestor.load_sample_emails()

    if emails:
        # Display sample emails
        ingestor.display_emails(emails)

        # Convert to DataFrame for analysis
        df = ingestor.convert_to_dataframe(emails)

        print(f"\n📊 DataFrame shape: {df.shape}")
        print("✅ Email ingestion system working correctly!")
    else:
        print("❌ No emails loaded. Please check your data file.")
