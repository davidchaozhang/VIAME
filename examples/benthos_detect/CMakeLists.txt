###
# Application level pipelines
##
set( BENTHOS_DETECT_EX_SOURCE
     benthos_detect.pipe
     input_list.txt )

if( WIN32 )
  set( BENTHOS_DETECT_EX_SOURCE
       ${BENTHOS_DETECT_EX_SOURCE}
       run_benthos_detect.bat )
else()
  set( BENTHOS_DETECT_EX_SOURCE
       ${BENTHOS_DETECT_EX_SOURCE}
       run_benthos_detect.sh )
endif()

###
# Install pipelines to example directory
##
install( FILES       ${BENTHOS_DETECT_EX_SOURCE}
         DESTINATION examples/benthos_detect )
