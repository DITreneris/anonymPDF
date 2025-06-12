#!/usr/bin/env python3
import os
import shutil
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/maintenance.log'),
        logging.StreamHandler()
    ]
)

class DocumentMaintenance:
    def __init__(self):
        self.temp_dir = Path('temp')
        self.processed_dir = Path('processed')
        self.logs_dir = Path('logs')
        self.uploads_dir = Path('uploads')
        
        # Create directories if they don't exist
        for directory in [self.temp_dir, self.processed_dir, self.logs_dir, self.uploads_dir]:
            directory.mkdir(exist_ok=True)
    
    def cleanup_temp_files(self, days=7):
        """Clean up temporary files older than specified days."""
        logging.info(f"Cleaning up temporary files older than {days} days...")
        cutoff_date = datetime.now() - timedelta(days=days)
        
        for file_path in self.temp_dir.glob('*'):
            if file_path.is_file():
                file_age = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_age < cutoff_date:
                    try:
                        file_path.unlink()
                        logging.info(f"Deleted temporary file: {file_path}")
                    except Exception as e:
                        logging.error(f"Error deleting {file_path}: {e}")
    
    def archive_processed_files(self, days=30):
        """Archive processed files older than specified days."""
        logging.info(f"Archiving processed files older than {days} days...")
        cutoff_date = datetime.now() - timedelta(days=days)
        archive_dir = self.processed_dir / 'archive'
        archive_dir.mkdir(exist_ok=True)
        
        for file_path in self.processed_dir.glob('*'):
            if file_path.is_file() and file_path.parent != archive_dir:
                file_age = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_age < cutoff_date:
                    try:
                        # Create year/month subdirectories
                        year_month = file_age.strftime('%Y-%m')
                        target_dir = archive_dir / year_month
                        target_dir.mkdir(exist_ok=True)
                        
                        # Move file to archive
                        shutil.move(str(file_path), str(target_dir / file_path.name))
                        logging.info(f"Archived {file_path} to {target_dir}")
                    except Exception as e:
                        logging.error(f"Error archiving {file_path}: {e}")
    
    def rotate_logs(self, max_size_mb=10):
        """Rotate log files that exceed the maximum size."""
        logging.info("Rotating log files...")
        max_size_bytes = max_size_mb * 1024 * 1024
        
        for log_file in self.logs_dir.glob('*.log'):
            if log_file.stat().st_size > max_size_bytes:
                try:
                    # Create backup with timestamp
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    backup_name = f"{log_file.stem}_{timestamp}{log_file.suffix}"
                    backup_path = log_file.parent / backup_name
                    
                    # Move current log to backup
                    shutil.move(str(log_file), str(backup_path))
                    
                    # Create new empty log file
                    log_file.touch()
                    
                    logging.info(f"Rotated log file: {log_file} -> {backup_path}")
                except Exception as e:
                    logging.error(f"Error rotating log file {log_file}: {e}")
    
    def cleanup_uploads(self, days=1):
        """Clean up uploaded files older than specified days."""
        logging.info(f"Cleaning up uploaded files older than {days} days...")
        cutoff_date = datetime.now() - timedelta(days=days)
        
        for file_path in self.uploads_dir.glob('*'):
            if file_path.is_file():
                file_age = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_age < cutoff_date:
                    try:
                        file_path.unlink()
                        logging.info(f"Deleted uploaded file: {file_path}")
                    except Exception as e:
                        logging.error(f"Error deleting {file_path}: {e}")
    
    def run_maintenance(self):
        """Run all maintenance tasks."""
        try:
            self.cleanup_temp_files()
            self.archive_processed_files()
            self.rotate_logs()
            self.cleanup_uploads()
            logging.info("Maintenance tasks completed successfully")
        except Exception as e:
            logging.error(f"Error during maintenance: {e}")

def main():
    """Main function to run maintenance tasks."""
    maintenance = DocumentMaintenance()
    maintenance.run_maintenance()

if __name__ == '__main__':
    main() 