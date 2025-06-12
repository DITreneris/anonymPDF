#!/usr/bin/env python3
"""
Priority 3 Setup Script
Automated environment preparation for Priority 3 ML implementation
"""

import os
import sys
import subprocess
import yaml
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} is compatible")
    return True

def create_directory_structure():
    """Create necessary directory structure for Priority 3."""
    print("📁 Creating directory structure...")
    
    directories = [
        "config",
        "app/core",
        "models",
        "data/training",
        "data/cache",
        "logs/ml",
        "tests/test_ml"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}")

def install_dependencies():
    """Install Priority 3 dependencies."""
    print("📦 Installing Priority 3 dependencies...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "-r", "requirements_priority3.txt", "--upgrade"
        ])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def generate_ml_config():
    """Generate ML configuration file."""
    print("⚙️ Generating ML configuration...")
    
    ml_config = {
        'ml_engine': {
            'model_type': 'xgboost',
            'model_params': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42
            },
            'feature_selection': {
                'max_features': 50,
                'feature_importance_threshold': 0.01
            },
            'training': {
                'test_size': 0.2,
                'cv_folds': 5,
                'scoring': 'roc_auc'
            },
            'confidence_calibration': {
                'method': 'isotonic',
                'cv_folds': 3
            }
        },
        'feature_engineering': {
            'text_features': {
                'length_features': True,
                'character_diversity': True,
                'linguistic_features': True,
                'pattern_features': True
            },
            'context_features': {
                'window_size': 50,
                'position_features': True,
                'document_structure': True
            },
            'nlp_features': {
                'pos_tags': True,
                'named_entities': True,
                'dependency_parse': False  # Disabled for performance
            }
        }
    }
    
    with open('config/ml_config.yaml', 'w') as f:
        yaml.dump(ml_config, f, default_flow_style=False)
    print("✅ ML configuration generated")

def generate_performance_config():
    """Generate performance optimization configuration."""
    print("🚀 Generating performance configuration...")
    
    performance_config = {
        'parallel_processing': {
            'max_workers': 'auto',  # Will use cpu_count()
            'chunk_size': 'adaptive',  # Dynamic chunk sizing
            'batch_size': 10,
            'memory_limit_mb': 1024
        },
        'caching': {
            'pattern_cache_size': 1000,
            'model_cache_enabled': True,
            'result_cache_ttl': 3600,
            'cache_compression': True
        },
        'memory_optimization': {
            'gc_frequency': 100,  # Run GC every 100 documents
            'spacy_optimize': True,
            'large_document_streaming': True,
            'memory_monitoring': True
        }
    }
    
    with open('config/performance_config.yaml', 'w') as f:
        yaml.dump(performance_config, f, default_flow_style=False)
    print("✅ Performance configuration generated")

def generate_analytics_config():
    """Generate analytics and monitoring configuration."""
    print("📊 Generating analytics configuration...")
    
    analytics_config = {
        'quality_metrics': {
            'precision_threshold': 0.95,
            'recall_threshold': 0.90,
            'f1_threshold': 0.92,
            'confidence_correlation_threshold': 0.85
        },
        'monitoring': {
            'real_time_enabled': True,
            'alert_thresholds': {
                'processing_time_ms': 5000,
                'memory_usage_mb': 2048,
                'error_rate': 0.01
            },
            'metrics_retention_days': 30
        },
        'feedback_system': {
            'collection_enabled': True,
            'auto_retrain_threshold': 100,  # Retrain after 100 feedback items
            'feedback_weight': 0.3
        }
    }
    
    with open('config/analytics_config.yaml', 'w') as f:
        yaml.dump(analytics_config, f, default_flow_style=False)
    print("✅ Analytics configuration generated")

def generate_multilang_config():
    """Generate multi-language configuration."""
    print("🌍 Generating multi-language configuration...")
    
    multilang_config = {
        'supported_languages': {
            'lt': {'name': 'Lithuanian', 'model': 'lt_core_news_sm', 'priority': 1},
            'en': {'name': 'English', 'model': 'en_core_web_sm', 'priority': 2},
            'lv': {'name': 'Latvian', 'model': 'xx_core_web_sm', 'priority': 3},
            'et': {'name': 'Estonian', 'model': 'xx_core_web_sm', 'priority': 4},
            'pl': {'name': 'Polish', 'model': 'pl_core_news_sm', 'priority': 5},
            'de': {'name': 'German', 'model': 'de_core_news_sm', 'priority': 6},
            'fr': {'name': 'French', 'model': 'fr_core_news_sm', 'priority': 7}
        },
        'language_detection': {
            'confidence_threshold': 0.8,
            'fallback_language': 'en',
            'mixed_language_handling': True
        },
        'cross_language_validation': {
            'enabled': True,
            'similarity_threshold': 0.7
        }
    }
    
    with open('config/multilang_config.yaml', 'w') as f:
        yaml.dump(multilang_config, f, default_flow_style=False)
    print("✅ Multi-language configuration generated")

def verify_setup():
    """Verify the setup was successful."""
    print("🔍 Verifying setup...")
    
    checks = [
        ('config/ml_config.yaml', 'ML configuration'),
        ('config/performance_config.yaml', 'Performance configuration'),
        ('config/analytics_config.yaml', 'Analytics configuration'),
        ('config/multilang_config.yaml', 'Multi-language configuration'),
        ('app/core', 'Core directory'),
        ('models', 'Models directory'),
        ('data/training', 'Training data directory')
    ]
    
    all_good = True
    for path, description in checks:
        if Path(path).exists():
            print(f"✅ {description}: OK")
        else:
            print(f"❌ {description}: Missing")
            all_good = False
    
    return all_good

def main():
    """Main setup function."""
    print("🚀 Priority 3 Setup Starting...")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directory structure
    create_directory_structure()
    
    # Install dependencies
    if not install_dependencies():
        print("⚠️  Warning: Dependency installation failed. You may need to install manually.")
    
    # Generate configuration files
    generate_ml_config()
    generate_performance_config()
    generate_analytics_config()
    generate_multilang_config()
    
    # Verify setup
    if verify_setup():
        print("\n🎉 Priority 3 setup completed successfully!")
        print("\nNext steps:")
        print("1. Review and customize configuration files in config/")
        print("2. Begin implementation with Session 1: ML Foundation")
        print("3. Run: python -c 'import app.core.ml_engine' to test ML setup")
    else:
        print("\n❌ Setup incomplete. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 