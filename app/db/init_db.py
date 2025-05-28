from app.database import Base, engine
from app.models import pdf_document


def init_db():
    Base.metadata.create_all(bind=engine)


if __name__ == "__main__":
    print("Creating database tables...")
    init_db()
    print("Database tables created successfully!")
