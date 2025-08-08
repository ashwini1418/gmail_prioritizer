from typing import Dict, List, Set
import json
import os


class TagManager:
    def __init__(self):
        self.tag_config_file = 'data/tag_config.json'
        self.tag_colors = {
            'URGENT': '#FF4444',  # Red
            'WORK': '#4A90E2',  # Blue
            'PERSONAL': '#7ED321',  # Green
            'MEETING': '#F5A623',  # Orange
            'DEADLINE': '#FF6B6B',  # Light Red
            'INVOICE': '#9013FE',  # Purple
            'PROMOTIONAL': '#FFC107',  # Yellow
            'SPAM': '#757575',  # Gray
            'SECURITY': '#E91E63',  # Pink
            'SOCIAL': '#4CAF50',  # Light Green
            'PROJECT': '#2196F3',  # Light Blue
            'NEWSLETTER': '#FF9800'  # Amber
        }

        self.priority_colors = {
            'HIGH': '#FF4444',  # Red
            'MEDIUM': '#FF9800',  # Orange
            'LOW': '#4CAF50'  # Green
        }

        self.load_tag_config()

    def load_tag_config(self):
        """Load tag configuration from file"""
        try:
            if os.path.exists(self.tag_config_file):
                with open(self.tag_config_file, 'r') as file:
                    config = json.load(file)
                    self.tag_colors.update(config.get('tag_colors', {}))
                    self.priority_colors.update(config.get('priority_colors', {}))
                print("✅ Tag configuration loaded")
            else:
                self.save_tag_config()
        except Exception as e:
            print(f"⚠️  Error loading tag config: {e}")

    def save_tag_config(self):
        """Save current tag configuration"""
        config = {
            'tag_colors': self.tag_colors,
            'priority_colors': self.priority_colors
        }

        os.makedirs(os.path.dirname(self.tag_config_file), exist_ok=True)

        try:
            with open(self.tag_config_file, 'w') as file:
                json.dump(config, file, indent=2)
            print("✅ Tag configuration saved")
        except Exception as e:
            print(f"❌ Error saving tag config: {e}")

    def get_tag_color(self, tag: str) -> str:
        """Get color for a specific tag"""
        return self.tag_colors.get(tag.upper(), '#757575')  # Default gray

    def get_priority_color(self, priority: str) -> str:
        """Get color for priority level"""
        return self.priority_colors.get(priority.upper(), '#757575')

    def add_custom_tag(self, tag: str, color: str = '#757575'):
        """Add a new custom tag with color"""
        self.tag_colors[tag.upper()] = color
        self.save_tag_config()
        print(f"✅ Added custom tag: {tag} with color {color}")

    def get_all_tags(self) -> List[str]:
        """Get list of all available tags"""
        return list(self.tag_colors.keys())

    def format_tags_html(self, tags: List[str]) -> str:
        """Format tags as colored HTML badges"""
        if not tags:
            return ""

        html_badges = []
        for tag in tags:
            color = self.get_tag_color(tag)
            badge = f'<span class="tag-badge" style="background-color: {color}; color: white; padding: 2px 8px; border-radius: 12px; font-size: 12px; margin: 2px;">{tag}</span>'
            html_badges.append(badge)

        return ' '.join(html_badges)

    def format_priority_html(self, priority: str) -> str:
        """Format priority as colored badge"""
        color = self.get_priority_color(priority)
        return f'<span class="priority-badge" style="background-color: {color}; color: white; padding: 4px 12px; border-radius: 16px; font-weight: bold;">{priority}</span>'

    def get_tag_statistics(self, emails: List[Dict]) -> Dict:
        """Get statistics about tag usage"""
        if not emails:
            return {}

        tag_counts = {}
        total_emails = len(emails)

        for email in emails:
            email_tags = email.get('tags', []) or email.get('predicted_tags', [])
            for tag in email_tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        # Calculate percentages
        tag_stats = {}
        for tag, count in tag_counts.items():
            tag_stats[tag] = {
                'count': count,
                'percentage': (count / total_emails) * 100,
                'color': self.get_tag_color(tag)
            }

        return dict(sorted(tag_stats.items(), key=lambda x: x[1]['count'], reverse=True))


# Test the tag manager
if __name__ == "__main__":
    print("🏷️  Testing Tag Manager System")
    print("=" * 50)

    # Initialize tag manager
    tag_manager = TagManager()

    # Test tag colors
    print("📋 Available tags and colors:")
    for tag in tag_manager.get_all_tags()[:8]:  # Show first 8 tags
        color = tag_manager.get_tag_color(tag)
        print(f"  {tag}: {color}")

    # Test priority colors
    print("\n🎯 Priority colors:")
    for priority in ['HIGH', 'MEDIUM', 'LOW']:
        color = tag_manager.get_priority_color(priority)
        print(f"  {priority}: {color}")

    # Test HTML formatting
    sample_tags = ['URGENT', 'WORK', 'MEETING']
    html_tags = tag_manager.format_tags_html(sample_tags)
    print(f"\n🌈 HTML formatted tags: {html_tags}")

    priority_html = tag_manager.format_priority_html('HIGH')
    print(f"🌈 HTML formatted priority: {priority_html}")

    print("\n✅ Tag Manager system working!")
