###
# VIAME Benthos Detect Processes
##

sprokit_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
  viame/processes/benthos_detect
  __init__)

sprokit_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/ProcessImage_sri.py
  viame/processes/benthos_detect
  ProcessImage_sri)
