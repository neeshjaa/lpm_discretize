# src/foo/__init__.py

# Import
from .discretize import *

# Explicitly import those two for testing (it' seems Python is trying
# to be smart when running import * and not actually import internal
# functions. We still want to test them, though.
from .discretize import _is_number
from .discretize import _prefix_category_name
from .discretize import _readable_category_name
