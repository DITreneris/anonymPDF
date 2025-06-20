# debug_imports.py
# Run this script to see what's actually available in your performance module

try:
    import app.core.performance as perf
    print("✓ Successfully imported app.core.performance")
    print(f"Available attributes: {[attr for attr in dir(perf) if not attr.startswith('_')]}")
    
    # Check specifically for the missing imports
    missing_items = []
    expected_items = ['file_processing_metrics', 'performance_monitor']
    
    for item in expected_items:
        if hasattr(perf, item):
            print(f"✓ Found {item}: {type(getattr(perf, item))}")
        else:
            print(f"✗ Missing {item}")
            missing_items.append(item)
    
    if missing_items:
        print(f"\n❌ Missing items: {missing_items}")
        print("You need to add these to your app/core/performance.py file")
    else:
        print("\n✅ All expected items found!")
        
except ImportError as e:
    print(f"❌ Cannot import app.core.performance: {e}")
    print("Check if the file exists and has correct syntax")

except Exception as e:
    print(f"❌ Unexpected error: {e}")