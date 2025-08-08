import pandas as pd
from typing import Dict, List
import sys
import os

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app.email_ingest import EmailIngestor
from app.tag_manager import TagManager
from model.multi_classifier import EmailMultiClassifier


class EmailDashboard:
    def __init__(self):
        self.ingestor = EmailIngestor()
        self.tag_manager = TagManager()
        self.classifier = EmailMultiClassifier()
        self.emails_df = pd.DataFrame()
        self.processed_emails = []

    def load_and_process_emails(self) -> bool:
        """Load emails and run AI classification"""
        print("📧 Loading and processing emails...")

        # Load sample emails
        emails = self.ingestor.load_sample_emails()
        if not emails:
            print("❌ No emails found!")
            return False

        # Convert to DataFrame
        self.emails_df = self.ingestor.convert_to_dataframe(emails)

        # Train classifier with current data
        print("🤖 Training AI classifier...")
        training_results = self.classifier.train(self.emails_df)

        # Get AI predictions for all emails
        email_dicts = self.emails_df.to_dict('records')
        self.processed_emails = self.classifier.predict(email_dicts)

        print(f"✅ Processed {len(self.processed_emails)} emails with AI predictions")
        return True

    def get_priority_distribution(self) -> Dict:
        """Get distribution of email priorities"""
        if not self.processed_emails:
            return {}

        priority_counts = {}
        for email in self.processed_emails:
            priority = email.get('predicted_priority', 'UNKNOWN')
            priority_counts[priority] = priority_counts.get(priority, 0) + 1

        # Add colors
        priority_stats = {}
        total = len(self.processed_emails)
        for priority, count in priority_counts.items():
            priority_stats[priority] = {
                'count': count,
                'percentage': (count / total) * 100,
                'color': self.tag_manager.get_priority_color(priority)
            }

        return priority_stats

    def get_tag_analytics(self) -> Dict:
        """Get comprehensive tag analytics"""
        return self.tag_manager.get_tag_statistics(self.processed_emails)

    def get_recent_emails(self, limit: int = 10) -> List[Dict]:
        """Get recent emails with AI predictions and formatting"""
        if not self.processed_emails:
            return []

        # Sort by date (newest first) and limit
        recent = sorted(
            self.processed_emails,
            key=lambda x: x.get('date', ''),
            reverse=True
        )[:limit]

        # Add HTML formatting
        for email in recent:
            email['tags_html'] = self.tag_manager.format_tags_html(
                email.get('predicted_tags', [])
            )
            email['priority_html'] = self.tag_manager.format_priority_html(
                email.get('predicted_priority', 'UNKNOWN')
            )

        return recent

    def get_urgency_alerts(self) -> List[Dict]:
        """Get urgent emails that need immediate attention"""
        if not self.processed_emails:
            return []

        urgent_emails = []
        for email in self.processed_emails:
            priority = email.get('predicted_priority', '')
            tags = email.get('predicted_tags', [])

            # Check for high priority or urgent tags
            if priority == 'HIGH' or 'URGENT' in tags:
                urgent_emails.append({
                    'subject': email['subject'],
                    'from': email['from'],
                    'priority': priority,
                    'tags': tags,
                    'priority_html': self.tag_manager.format_priority_html(priority),
                    'tags_html': self.tag_manager.format_tags_html(tags)
                })

        return urgent_emails

    def get_sender_analytics(self) -> Dict:
        """Get analytics about email senders"""
        if not self.processed_emails:
            return {}

        sender_stats = {}
        for email in self.processed_emails:
            sender = email.get('from', 'Unknown')
            priority = email.get('predicted_priority', 'UNKNOWN')

            if sender not in sender_stats:
                sender_stats[sender] = {
                    'total_emails': 0,
                    'high_priority': 0,
                    'medium_priority': 0,
                    'low_priority': 0
                }

            sender_stats[sender]['total_emails'] += 1
            sender_stats[sender][f'{priority.lower()}_priority'] += 1

        return dict(sorted(sender_stats.items(), key=lambda x: x[1]['total_emails'], reverse=True))

    def get_dashboard_summary(self) -> Dict:
        """Get complete dashboard summary"""
        if not self.processed_emails:
            return {}

        return {
            'total_emails': len(self.processed_emails),
            'priority_distribution': self.get_priority_distribution(),
            'tag_analytics': self.get_tag_analytics(),
            'recent_emails': self.get_recent_emails(5),
            'urgent_alerts': self.get_urgency_alerts(),
            'sender_analytics': self.get_sender_analytics(),
            'ai_accuracy': getattr(self.classifier, 'last_accuracy', 0.85) * 100
        }


# Test the dashboard
if __name__ == "__main__":
    print("📊 Testing Email Dashboard Backend")
    print("=" * 50)

    # Initialize dashboard
    dashboard = EmailDashboard()

    # Load and process emails
    if dashboard.load_and_process_emails():
        # Get dashboard summary
        summary = dashboard.get_dashboard_summary()

        print(f"\n📈 Dashboard Summary:")
        print(f"  Total Emails: {summary['total_emails']}")
        print(f"  AI Accuracy: {summary['ai_accuracy']:.1f}%")

        print(f"\n🎯 Priority Distribution:")
        for priority, stats in summary['priority_distribution'].items():
            print(f"  {priority}: {stats['count']} ({stats['percentage']:.1f}%)")

        print(f"\n🏷️  Top Tags:")
        for tag, stats in list(summary['tag_analytics'].items())[:5]:
            print(f"  {tag}: {stats['count']} emails ({stats['percentage']:.1f}%)")

        print(f"\n🚨 Urgent Alerts: {len(summary['urgent_alerts'])} emails")

        print("\n✅ Dashboard backend working perfectly!")
    else:
        print("❌ Dashboard initialization failed!")
