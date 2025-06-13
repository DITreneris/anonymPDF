# This script is used to delete the plotly directory from the site-packages directory

import os
import shutil

site_packages = os.path.join(
    os.environ['VIRTUAL_ENV'],
    'Lib',
    'site-packages'
)

for item in os.listdir(site_packages):
    if item.lower().startswith("~lotly"):
        full_path = os.path.join(site_packages, item)
        print(f"Deleting: {full_path}")
        if os.path.isdir(full_path):
            shutil.rmtree(full_path)
        else:
            os.remove(full_path)
