from flask import Flask, render_template, jsonify, request, redirect, url_for
import os
import sys
import json
from datetime import datetime

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app.dashboard import EmailDashboard
from app.tag_manager import TagManager
from model.multi_classifier import EmailMultiClassifier

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this in production

# Initialize dashboard components
dashboard = EmailDashboard()
tag_manager = TagManager()
classifier = EmailMultiClassifier()

# Load and process emails on startup
print("🚀 Starting Email Prioritizer Web Application...")
dashboard.load_and_process_emails()
print("✅ Email Prioritizer ready!")


@app.route('/')
def index():
    """Main dashboard page"""
    try:
        # Get dashboard summary
        summary = dashboard.get_dashboard_summary()

        return render_template('dashboard.html',
                               summary=summary,
                               page_title="Email Prioritizer Dashboard")
    except Exception as e:
        print(f"❌ Error loading dashboard: {e}")
        return render_template('error.html', error=str(e))


@app.route('/emails')
def emails_list():
    """Email list page with AI predictions"""
    try:
        # Get all processed emails
        emails = dashboard.get_recent_emails(20)  # Show more emails

        return render_template('emails.html',
                               emails=emails,
                               page_title="Email List")
    except Exception as e:
        print(f"❌ Error loading emails: {e}")
        return render_template('error.html', error=str(e))


@app.route('/analytics')
def analytics():
    """Analytics page with detailed statistics"""
    try:
        summary = dashboard.get_dashboard_summary()

        return render_template('analytics.html',
                               summary=summary,
                               page_title="Email Analytics")
    except Exception as e:
        print(f"❌ Error loading analytics: {e}")
        return render_template('error.html', error=str(e))


@app.route('/api/priority-distribution')
def api_priority_distribution():
    """API endpoint for priority distribution data"""
    try:
        distribution = dashboard.get_priority_distribution()
        return jsonify(distribution)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/tag-analytics')
def api_tag_analytics():
    """API endpoint for tag analytics data"""
    try:
        analytics = dashboard.get_tag_analytics()
        return jsonify(analytics)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/classify-email', methods=['POST'])
def api_classify_email():
    """API endpoint to classify a new email"""
    try:
        email_data = request.get_json()

        # Validate required fields
        required_fields = ['subject', 'body', 'from']
        if not all(field in email_data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400

        # Classify the email
        prediction = classifier.predict([email_data])

        if prediction:
            result = prediction[0]
            return jsonify({
                'predicted_priority': result.get('predicted_priority'),
                'predicted_tags': result.get('predicted_tags'),
                'priority_html': tag_manager.format_priority_html(
                    result.get('predicted_priority', 'UNKNOWN')
                ),
                'tags_html': tag_manager.format_tags_html(
                    result.get('predicted_tags', [])
                )
            })
        else:
            return jsonify({'error': 'Classification failed'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/test-email')
def test_email():
    """Test page for trying email classification"""
    return render_template('test_email.html', page_title="Test Email Classification")


@app.errorhandler(404)
def not_found(error):
    return render_template('error.html',
                           error="Page not found",
                           page_title="404 Error"), 404


@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html',
                           error="Internal server error",
                           page_title="500 Error"), 500


if __name__ == '__main__':
    print("🌐 Starting Flask development server...")
    print("📧 Email Prioritizer will be available at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
