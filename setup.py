from setuptools import setup, find_packages

setup(
    name="anonympdf",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi==0.104.1",
        "uvicorn==0.24.0",
        "python-multipart==0.0.6",
        "PyPDF2==3.0.1",
        "pdfminer.six==20221105",
        "spacy==3.7.2",
        "sqlalchemy==2.0.23",
        "python-jose==3.3.0",
        "pydantic==2.5.2",
        "python-dotenv==1.0.0"
    ],
) 