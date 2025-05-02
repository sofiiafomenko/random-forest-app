
import os
import sys

# Add the current working directory to the system path
# This allows us to import modules from the current directory
sys.path.append(os.getcwd()+'/decisiontrees')

from treeclassifier import *
from treeregressor import *