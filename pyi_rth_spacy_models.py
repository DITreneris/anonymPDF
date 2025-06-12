# PyInstaller runtime hook for spaCy models
# This hook sets up the environment for spaCy models in PyInstaller bundles

import sys
import os
from pathlib import Path

def setup_spacy_models():
    """Setup spaCy model paths for PyInstaller environment"""
    
    # Get bundle directory
    if hasattr(sys, '_MEIPASS'):
        bundle_dir = Path(sys._MEIPASS)
        print(f"spaCy models setup completed. Bundle dir: {bundle_dir}")
        
        # Define model locations to check
        model_locations = [
            # Direct bundle locations
            bundle_dir / 'en_core_web_sm',
            bundle_dir / 'lt_core_news_sm',
            # Site-packages style locations
            bundle_dir / 'site-packages' / 'en_core_web_sm',
            bundle_dir / 'site-packages' / 'lt_core_news_sm',
        ]
        
        # Add model paths to sys.path if they exist
        for model_path in model_locations:
            if model_path.exists():
                sys.path.insert(0, str(model_path.parent))
                print(f"Added to sys.path: {model_path.parent}")
                
                # Also set environment variable for direct model access
                model_name = model_path.name
                env_var = f'SPACY_MODEL_{model_name.upper()}'
                os.environ[env_var] = str(model_path)
                print(f"Set environment variable {env_var} = {model_path}")
        
        # Set general spaCy data path
        os.environ['SPACY_DATA'] = str(bundle_dir)
        
        # Alternative: add bundle directory to Python path for model discovery
        if str(bundle_dir) not in sys.path:
            sys.path.insert(0, str(bundle_dir))
            
        # Set up spaCy to look in bundle directory
        try:
            import spacy.util
            # Monkey-patch get_package_path to check bundle first
            original_get_package_path = spacy.util.get_package_path
            
            def patched_get_package_path(name):
                """Check bundle directory first, then fall back to original"""
                for model_path in model_locations:
                    if model_path.name == name and model_path.exists():
                        return model_path
                return original_get_package_path(name)
            
            spacy.util.get_package_path = patched_get_package_path
            print("spaCy get_package_path patched for PyInstaller")
            
        except ImportError:
            pass  # spaCy not available yet
            
    else:
        # Not in PyInstaller bundle, normal operation
        print("spaCy models setup: Not in PyInstaller bundle, using standard paths")

# Execute the setup
setup_spacy_models() 