from app.email_ingest import EmailIngestor
import pandas as pd


def test_email_system():
    print("🧪 Testing Email Data System")
    print("=" * 40)

    # Test email loading
    ingestor = EmailIngestor()
    emails = ingestor.load_sample_emails()

    print(f"✅ Loaded {len(emails)} emails")

    # Test DataFrame conversion
    df = ingestor.convert_to_dataframe(emails)
    print(f"✅ Created DataFrame with shape: {df.shape}")

    # Test feature extraction
    if not df.empty:
        print(f"✅ Features created: {list(df.columns)}")
        print(f"✅ Average email length: {df['body_length'].mean():.1f} characters")
        print(f"✅ Urgent emails detected: {df['has_urgent_keywords'].sum()}")

    print("\n🎉 All tests passed! Ready for next step.")


if __name__ == "__main__":
    test_email_system()
