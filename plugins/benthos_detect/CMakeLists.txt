###
# VIAME Benthos Detect Process
##

find_package( PythonInterp REQUIRED )
find_package( PythonLibs REQUIRED )

set(kwiver_python_subdir "python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")
set(kwiver_python_output_path "${VIAME_BUILD_PREFIX}/lib/${kwiver_python_subdir}")

kwiver_create_python_plugin_init(kwiver)
kwiver_create_python_plugin_init(kwiver/processes)

# add python processes here

kwiver_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/ProcessImage_sri.py
  kwiver/processes
  ProcessImage_sri)
