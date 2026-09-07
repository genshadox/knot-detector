"""
pytest configuration for the tests/ directory.

tests/ lives outside src/ so we need to add src/ to sys.path to allow imports from it.
"""
 
import sys
from pathlib import Path
 
SRC_DIR = Path(__file__).parent.parent / "src" # relative path to the src/ directory
sys.path.insert(0, str(SRC_DIR))