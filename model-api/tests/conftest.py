import os
import sys
import pytest

# Set default environment variables for testing
os.environ["GITHUB_REPO"] = "org/repo"
os.environ["GITHUB_TOKEN"] = "ghp_test"

# Pre-import config module so it's in sys.modules for the tests
import config
