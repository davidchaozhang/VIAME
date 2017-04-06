###
# VIAME Benthos Detect Processes
##

kwiver_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
  viame/processes/benthos_detect
  __init__)

kwiver_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/ProcessImage_sri.py
  viame/processes/benthos_detect
  ProcessImage_sri)
