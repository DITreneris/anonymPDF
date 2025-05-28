from sqlalchemy import text, inspect
from sqlalchemy.orm import Session
from app.database import engine, get_db
from app.core.logging import db_logger
from typing import Dict, Any
import json


class DatabaseMigration:
    """Handles database schema migrations for AnonymPDF."""

    def __init__(self):
        self.migrations = [
            self.migration_001_add_redaction_report_fields,
        ]

    def check_table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database."""
        inspector = inspect(engine)
        return table_name in inspector.get_table_names()

    def check_column_exists(self, table_name: str, column_name: str) -> bool:
        """Check if a column exists in a table."""
        inspector = inspect(engine)
        if not self.check_table_exists(table_name):
            return False

        columns = [col["name"] for col in inspector.get_columns(table_name)]
        return column_name in columns

    def migration_001_add_redaction_report_fields(self, db: Session) -> bool:
        """
        Migration 001: Add redaction_report and processing_metadata columns
        to separate error messages from redaction reports.
        """
        migration_name = "001_add_redaction_report_fields"
        db_logger.info(f"Running migration: {migration_name}")

        try:
            # Check if columns already exist
            if self.check_column_exists(
                "pdf_documents", "redaction_report"
            ) and self.check_column_exists("pdf_documents", "processing_metadata"):
                db_logger.info(f"Migration {migration_name} already applied")
                return True

            # Add redaction_report column
            if not self.check_column_exists("pdf_documents", "redaction_report"):
                db.execute(text("ALTER TABLE pdf_documents ADD COLUMN redaction_report TEXT"))
                db_logger.log_database_operation(
                    operation="add_column", table="pdf_documents", column="redaction_report"
                )

            # Add processing_metadata column
            if not self.check_column_exists("pdf_documents", "processing_metadata"):
                # SQLite doesn't support JSON type, use TEXT instead
                db.execute(text("ALTER TABLE pdf_documents ADD COLUMN processing_metadata TEXT"))
                db_logger.log_database_operation(
                    operation="add_column", table="pdf_documents", column="processing_metadata"
                )

            # Migrate existing data: move reports from error_message to redaction_report
            result = db.execute(
                text(
                    """
                SELECT id, error_message 
                FROM pdf_documents 
                WHERE error_message IS NOT NULL 
                AND status = 'completed'
            """
                )
            )

            migrated_count = 0
            for row in result:
                doc_id, error_message = row

                # Check if error_message looks like a JSON report
                try:
                    # Try to parse as JSON to see if it's a report
                    json.loads(error_message)

                    # Move to redaction_report and clear error_message
                    db.execute(
                        text(
                            """
                        UPDATE pdf_documents 
                        SET redaction_report = :report, error_message = NULL 
                        WHERE id = :doc_id
                    """
                        ),
                        {"report": error_message, "doc_id": doc_id},
                    )

                    migrated_count += 1

                except json.JSONDecodeError:
                    # Keep as error_message if it's not valid JSON
                    pass

            db.commit()

            db_logger.info(
                f"Migration {migration_name} completed successfully. "
                f"Migrated {migrated_count} records."
            )
            return True

        except Exception as e:
            db.rollback()
            db_logger.log_error(f"migration_{migration_name}", e)
            return False

    def run_migrations(self) -> bool:
        """Run all pending migrations."""
        db_logger.info("Starting database migrations")

        # Get database session
        db = next(get_db())

        try:
            success_count = 0
            for i, migration in enumerate(self.migrations):
                migration_number = i + 1
                db_logger.info(f"Running migration {migration_number}/{len(self.migrations)}")

                if migration(db):
                    success_count += 1
                else:
                    db_logger.error(f"Migration {migration_number} failed")
                    return False

            db_logger.info(f"All {success_count} migrations completed successfully")
            return True

        except Exception as e:
            db_logger.log_error("run_migrations", e)
            return False
        finally:
            db.close()

    def initialize_database(self) -> bool:
        """Initialize database with all tables and run migrations."""
        db_logger.info("Initializing database")

        try:
            # Import models to ensure they're registered
            from app.models.pdf_document import PDFDocument
            from app.database import Base

            # Create all tables
            Base.metadata.create_all(bind=engine)
            db_logger.log_database_operation(operation="create_tables", table="all")

            # Run migrations
            if self.run_migrations():
                db_logger.info("Database initialization completed successfully")
                return True
            else:
                db_logger.error("Database initialization failed during migrations")
                return False

        except Exception as e:
            db_logger.log_error("initialize_database", e)
            return False

    def get_database_info(self) -> Dict[str, Any]:
        """Get information about the current database state."""
        try:
            inspector = inspect(engine)

            info = {"tables": inspector.get_table_names(), "pdf_documents_columns": []}

            if "pdf_documents" in info["tables"]:
                columns = inspector.get_columns("pdf_documents")
                info["pdf_documents_columns"] = [
                    {"name": col["name"], "type": str(col["type"]), "nullable": col["nullable"]}
                    for col in columns
                ]

            return info

        except Exception as e:
            db_logger.log_error("get_database_info", e)
            return {"error": str(e)}


def initialize_database_on_startup():
    """Function to be called on application startup to initialize database."""
    migration_manager = DatabaseMigration()

    if migration_manager.initialize_database():
        db_logger.info("Database startup initialization successful")
        return True
    else:
        db_logger.error("Database startup initialization failed")
        return False


# Global migration manager instance
migration_manager = DatabaseMigration()
