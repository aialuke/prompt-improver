import json
from git import Repo
from pathlib import Path

# File paths
roadmap_delta_path = Path('roadmap_delta.json')
roadmap_md_path = Path('MCP_ROADMAP.md')

# Read roadmap_delta.json
with open(roadmap_delta_path, 'r', encoding='utf-8') as f:
    roadmap_delta = json.load(f)

# Construct sections to add
progress_tracker = """
⏳ Progress Tracker
==================
Overall Implementation Coverage: {implementation_coverage}
Completed: {completed_items} of {total_roadmap_items}
Missing: {missing_items}
""".format(
    implementation_coverage=roadmap_delta['analysis_metadata']['implementation_coverage'],
    completed_items=roadmap_delta['analysis_metadata']['completed_items'],
    total_roadmap_items=roadmap_delta['analysis_metadata']['total_roadmap_items'],
    missing_items=roadmap_delta['analysis_metadata']['missing_items'],
)

integration_guide = """
🔗 Integration Guide
===================
To integrate the `send_training_batch` function with the real ML pipeline once it becomes available, it's essential to replace the current stub implementation found in the `ml_integration.py` with the actual ML endpoint integration code. Ensure robust error handling and compatibility with the expected data format.
"""

# Read current MCP_ROADMAP.md
with open(roadmap_md_path, 'r', encoding='utf-8') as f:
    current_roadmap_content = f.read()

# Write new roadmap with additions at the top
new_roadmap_content = progress_tracker + integration_guide + current_roadmap_content

with open(roadmap_md_path, 'w', encoding='utf-8') as f:
    f.write(new_roadmap_content)

# Initiate repo
repo = Repo('.')

# Check and commit changes
repo.git.add('MCP_ROADMAP.md')
diff = repo.git.diff('MCP_ROADMAP.md', cached=True)

if diff:
    repo.index.commit('Update MCP_ROADMAP.md with progress tracker and integration guide')
