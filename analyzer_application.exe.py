#!/usr/bin/env python3
"""
IPMDAR Analyzer Application Executable Wrapper
This script serves as the entry point for the analyzer application executable.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import and run the main application
from main import main

if __name__ == '__main__':
    main()
