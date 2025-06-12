# -*- mode: python ; coding: utf-8 -*-
# AnonymPDF PyInstaller Specification File
# For Windows Deployment

import sys
import os
from pathlib import Path

# Get the current directory
WORK_DIR = Path.cwd()

# Define application metadata
APP_NAME = "AnonymPDF"
APP_VERSION = "2.1.1"

block_cipher = None

# Function to get spaCy model paths - FIXED
def get_spacy_model_path(model_name):
    try:
        import spacy.util
        return spacy.util.get_package_path(model_name)
    except Exception as e:
        print(f"Could not find spaCy model {model_name}: {e}")
        return None

# Get spaCy model paths
en_model_path = get_spacy_model_path("en_core_web_sm")
lt_model_path = get_spacy_model_path("lt_core_news_sm")

# Define data files to include
added_files = [
    # Configuration files
    ('config', 'config'),
    # Frontend built assets
    ('frontend/dist', 'frontend/dist'),
    # README and documentation
    ('README.md', '.'),
]

# Add spaCy models if found - FIXED BUNDLING
if en_model_path:
    # Bundle the entire model directory to the correct location
    added_files.append((str(en_model_path), 'en_core_web_sm'))
    print(f"Adding English spaCy model from: {en_model_path}")

if lt_model_path:
    # Bundle the entire model directory to the correct location
    added_files.append((str(lt_model_path), 'lt_core_news_sm'))
    print(f"Adding Lithuanian spaCy model from: {lt_model_path}")

# Also add model data explicitly for PyInstaller to find
try:
    import spacy
    # Add the models as packages to ensure they're bundled
    if en_model_path and os.path.exists(en_model_path):
        added_files.append((str(en_model_path) + "/*", 'site-packages/en_core_web_sm/'))
    if lt_model_path and os.path.exists(lt_model_path):
        added_files.append((str(lt_model_path) + "/*", 'site-packages/lt_core_news_sm/'))
except Exception as e:
    print(f"Warning: Could not add spaCy models as packages: {e}")

# Define hidden imports (packages not automatically detected)
hidden_imports = [
    # Core application modules
    'app.main',
    'app.core.dependencies',
    'app.core.logging',
    'app.db.migrations',
    'app.api.endpoints.pdf',
    'app.services.pdf_processor',
    'app.models',
    'app.schemas',
    
    # spaCy and language models - enhanced imports
    'spacy',
    'spacy.cli',
    'spacy.util',
    'spacy.lang.en',
    'spacy.lang.lt',
    'spacy.pipeline',
    'spacy.tokens',
    'spacy.vocab',
    'spacy.strings',
    'spacy.matcher',
    'spacy.lang.en.lex_attrs',
    'spacy.lang.en.stop_words',
    'spacy.lang.lt.lex_attrs',
    'spacy.lang.lt.stop_words',
    
    # Model packages directly
    'en_core_web_sm',
    'lt_core_news_sm',
    
    # PDF processing libraries
    'fitz',  # PyMuPDF
    'PyPDF2',
    'pdfminer',
    'pdfminer.six',
    'pdfminer.layout',
    'pdfminer.high_level',
    'reportlab',
    'reportlab.pdfgen',
    'reportlab.lib',
    
    # FastAPI and dependencies
    'fastapi',
    'fastapi.applications',
    'fastapi.routing',
    'fastapi.middleware',
    'fastapi.middleware.cors',
    'uvicorn',
    'uvicorn.main',
    'uvicorn.server',
    'uvicorn.workers',
    'uvicorn.protocols',
    'uvicorn.protocols.http',
    'uvicorn.protocols.websockets',
    'starlette',
    'starlette.applications',
    'starlette.middleware',
    'starlette.routing',
    'starlette.responses',
    
    # SQLAlchemy and database
    'sqlalchemy',
    'sqlalchemy.dialects',
    'sqlalchemy.dialects.sqlite',
    'sqlmodel',
    'alembic',
    'aiosqlite',
    
    # ML and NLP libraries
    'sklearn',
    'sklearn.ensemble',
    'sklearn.feature_extraction',
    'sklearn.feature_extraction.text',
    'sklearn.metrics',
    'sklearn.model_selection',
    'xgboost',
    'numpy',
    'numpy.core',
    'numpy.lib',
    'pandas',
    'pandas.core',
    
    # Language detection
    'langdetect',
    'langdetect.detector',
    'langdetect.lang_detect_exception',
    'nltk',
    'nltk.corpus',
    'nltk.tokenize',
    'textblob',
    
    # Configuration and utilities
    'yaml',
    'pydantic',
    'pydantic_settings',
    'dotenv',
    'multiprocessing',
    'asyncio',
    'aiofiles',
    'aiohttp',
    
    # Performance and monitoring
    'psutil',
    'memory_profiler',
    'cachetools',
    
    # Serialization
    'pickle',
    'joblib',
    'dill',
    
    # Additional dependencies that might be missed
    'typing_extensions',
    'email_validator',
    'python_multipart',
]

# Binaries to exclude (problematic on Windows)
excludes = [
    'tkinter',
    'matplotlib.backends._backend_tk',
    'PIL.ImageTk',
    'tornado',
    'zmq',
    'IPython',
    'jupyter',
    'notebook',
    'sphinx',
    'py-spy',  # Windows compatibility issues
    'polyglot',  # Windows compilation issues
    'pycld2',  # Windows compilation issues
]

a = Analysis(
    ['app/main.py'],
    pathex=[str(WORK_DIR)],
    binaries=[],
    datas=added_files,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['pyi_rth_spacy_models.py'],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Remove duplicate entries
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Set to False for windowed app, True for debugging
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # Windows-specific options
    version_file=None,  # We'll create this later
    icon='assets/anonympdf.ico' if Path('assets/anonympdf.ico').exists() else None,
)

# Create distribution directory
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=APP_NAME,
) 