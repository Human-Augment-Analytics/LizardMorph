# Place this file in your backend directory
# A simple script to verify that all imports work correctly

import sys
import os

print("Python version:", sys.version)
print("Current directory:", os.getcwd())
print("Contents of current directory:", os.listdir('.'))

print("\nTrying to import all required modules...")

try:
    import numpy
    print("✓ NumPy imported successfully:", numpy.__version__)
except Exception as e:
    print("✗ Failed to import NumPy:", str(e))

try:
    import pandas
    print("✓ Pandas imported successfully:", pandas.__version__)
except Exception as e:
    print("✗ Failed to import Pandas:", str(e))

try:
    import cv2
    print("✓ OpenCV imported successfully:", cv2.__version__)
except Exception as e:
    print("✗ Failed to import OpenCV:", str(e))

try:
    import dlib
    print("✓ dlib imported successfully:", dlib.__version__)
except Exception as e:
    print("✗ Failed to import dlib:", str(e))

try:
    import matplotlib
    print("✓ Matplotlib imported successfully:", matplotlib.__version__)
except Exception as e:
    print("✗ Failed to import Matplotlib:", str(e))

try:
    from PIL import Image
    import PIL
    print("✓ Pillow imported successfully:", PIL.__version__)
except Exception as e:
    print("✗ Failed to import Pillow:", str(e))

try:
    import flask
    print("✓ Flask imported successfully:", flask.__version__)
except Exception as e:
    print("✗ Failed to import Flask:", str(e))

try:
    import flask_cors
    print("✓ Flask-CORS imported successfully:", flask_cors.__version__)
except Exception as e:
    print("✗ Failed to import Flask-CORS:", str(e))

try:
    import gunicorn
    print("✓ Gunicorn imported successfully:", gunicorn.__version__)
except Exception as e:
    print("✗ Failed to import Gunicorn:", str(e))

try:
    import pydicom
    print("✓ Pydicom imported successfully:", pydicom.__version__)
except Exception as e:
    print("✗ Failed to import Pydicom:", str(e))

print("\nNow trying to import your application modules...")

try:
    import utils
    print("✓ utils module imported successfully")
except Exception as e:
    print("✗ Failed to import utils:", str(e))
    print("Stack trace:")
    import traceback
    traceback.print_exc()

print("\nEnd of diagnostics")