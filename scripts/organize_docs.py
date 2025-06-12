#!/usr/bin/env python3
import os
import shutil
from datetime import datetime
from pathlib import Path

def create_directory_structure():
    """Create the new directory structure for documentation."""
    directories = [
        'docs/api',
        'docs/development',
        'docs/deployment',
        'docs/user',
        'tests/unit',
        'tests/integration',
        'tests/data'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

def categorize_document(filename):
    """Categorize a document based on its name and content."""
    filename_lower = filename.lower()
    
    # API Documentation
    if 'api' in filename_lower:
        return 'docs/api'
    
    # Development Documentation
    dev_keywords = ['dev', 'development', 'implementation', 'code', 'technical']
    if any(keyword in filename_lower for keyword in dev_keywords):
        return 'docs/development'
    
    # Deployment Documentation
    deploy_keywords = ['deploy', 'deployment', 'setup', 'install', 'configuration']
    if any(keyword in filename_lower for keyword in deploy_keywords):
        return 'docs/deployment'
    
    # User Documentation
    user_keywords = ['user', 'guide', 'manual', 'tutorial', 'how-to']
    if any(keyword in filename_lower for keyword in user_keywords):
        return 'docs/user'
    
    # Default to development if unsure
    return 'docs/development'

def rename_file(filename):
    """Rename file according to naming conventions."""
    # Convert to lowercase and replace spaces with hyphens
    new_name = filename.lower().replace(' ', '-')
    
    # Add date prefix if it's a time-sensitive document
    time_sensitive_keywords = ['meeting', 'report', 'summary', 'notes']
    if any(keyword in new_name for keyword in time_sensitive_keywords):
        date_str = datetime.now().strftime('%Y-%m-%d')
        new_name = f"{date_str}-{new_name}"
    
    return new_name

def organize_documents():
    """Organize existing documents into the new structure."""
    # Get all markdown files in the root directory
    root_dir = Path('.')
    md_files = list(root_dir.glob('*.md'))
    
    for file_path in md_files:
        # Skip files that are already in the docs directory
        if 'docs/' in str(file_path):
            continue
        
        # Determine the target directory
        target_dir = categorize_document(file_path.name)
        
        # Create the new filename
        new_filename = rename_file(file_path.name)
        
        # Create the target path
        target_path = Path(target_dir) / new_filename
        
        # Move the file
        try:
            shutil.move(str(file_path), str(target_path))
            print(f"Moved {file_path} to {target_path}")
        except Exception as e:
            print(f"Error moving {file_path}: {e}")

def main():
    """Main function to organize the documentation."""
    print("Creating directory structure...")
    create_directory_structure()
    
    print("\nOrganizing documents...")
    organize_documents()
    
    print("\nDocument organization complete!")

if __name__ == '__main__':
    main() 