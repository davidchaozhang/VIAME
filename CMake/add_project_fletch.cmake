# fletch External Project
#
# Required symbols are:
#   VIAME_BUILD_PREFIX - where packages are built
#   VIAME_BUILD_INSTALL_PREFIX - directory install target
#   VIAME_PACKAGES_DIR - location of git submodule packages
#   VIAME_ARGS_COMMON -
##

set( VIAME_PROJECT_LIST ${VIAME_PROJECT_LIST} fletch )

if( VIAME_ENABLE_PYTHON )
  FormatPassdowns( "PYTHON" VIAME_PYTHON_FLAGS )
endif()

if( VIAME_ENABLE_CUDA )
  FormatPassdowns( "CUDA" VIAME_CUDA_FLAGS )
endif()

if( VIAME_ENABLE_CUDNN )
  FormatPassdowns( "CUDNN" VIAME_CUDNN_FLAGS )
endif()

if( VIAME_ENABLE_VXL OR VIAME_ENABLE_OPENCV )
  set( fletch_DEP_FLAGS
    ${fletch_DEP_FLAGS}
    -Dfletch_ENABLE_ZLib:BOOL=ON
    -Dfletch_ENABLE_libjpeg-turbo:BOOL=ON
    -Dfletch_ENABLE_libtiff:BOOL=ON
    -Dfletch_ENABLE_PNG:BOOL=ON
  )
endif()

if( VIAME_ENABLE_VIVIA )
  set( fletch_DEP_FLAGS
    ${fletch_DEP_FLAGS}
    -Dfletch_ENABLE_GeographicLib:BOOL=ON
    -Dfletch_ENABLE_TinyXML:BOOL=ON
    -Dfletch_ENABLE_shapelib:BOOL=ON
    -Dfletch_ENABLE_libjson:BOOL=ON
    -Dfletch_ENABLE_Qt:BOOL=ON
    -Dfletch_ENABLE_VTK:BOOL=ON
    -Dfletch_ENABLE_PROJ4:BOOL=ON
    -Dfletch_ENABLE_libkml:BOOL=ON
    -Dfletch_ENABLE_PNG:BOOL=ON
  )
  if( NOT WIN32 )
    set( fletch_DEP_FLAGS
      ${fletch_DEP_FLAGS}
      -Dfletch_ENABLE_libxml2:BOOL=ON
    )
  endif()
else()
  set( fletch_DEP_FLAGS
    ${fletch_DEP_FLAGS}
    -Dfletch_ENABLE_Qt:BOOL=OFF
    -Dfletch_ENABLE_VTK:BOOL=OFF
  )
endif()

if( VIAME_ENABLE_KWANT )
  set( fletch_DEP_FLAGS
    ${fletch_DEP_FLAGS}
    -Dfletch_ENABLE_TinyXML:BOOL=ON
    -Dfletch_ENABLE_libjson:BOOL=ON
  )
endif()

if( VIAME_ENABLE_CUDA )
  set( fletch_DEP_FLAGS
    ${fletch_DEP_FLAGS}
    -Dfletch_BUILD_WITH_CUDA:BOOL=ON
  )
  if( VIAME_ENABLE_CUDNN )
    set( fletch_DEP_FLAGS
      ${fletch_DEP_FLAGS}
      -Dfletch_BUILD_WITH_CUDNN:BOOL=ON
    )
  else()
    set( fletch_DEP_FLAGS
      ${fletch_DEP_FLAGS}
      -Dfletch_BUILD_WITH_CUDNN:BOOL=OFF
    )
  endif()
else()
  set( fletch_DEP_FLAGS
    ${fletch_DEP_FLAGS}
    -Dfletch_BUILD_WITH_CUDA:BOOL=OFF
  )
endif()

if( VIAME_ENABLE_FFMPEG )
  set( fletch_DEP_FLAGS
    ${fletch_DEP_FLAGS}
    -Dfletch_ENABLE_FFmpeg:BOOL=ON
  )
else()
  set( fletch_DEP_FLAGS
    ${fletch_DEP_FLAGS}
    -Dfletch_ENABLE_FFmpeg:BOOL=OFF
  )
endif()

ExternalProject_Add(fletch
  PREFIX ${VIAME_BUILD_PREFIX}
  SOURCE_DIR ${VIAME_PACKAGES_DIR}/fletch
  CMAKE_GENERATOR ${gen}
  CMAKE_ARGS
    ${VIAME_ARGS_COMMON}
    ${VIAME_PYTHON_FLAGS}
    ${VIAME_CUDA_FLAGS}
    ${VIAME_CUDNN_FLAGS}

    -DBUILD_SHARED_LIBS:BOOL=ON

    # KWIVER Dependencies, Always On
    -Dfletch_ENABLE_Boost:BOOL=TRUE
    -Dfletch_ENABLE_Eigen:BOOL=TRUE

    # Optional Dependencies
    ${fletch_DEP_FLAGS}

    -Dfletch_ENABLE_VXL:BOOL=${VIAME_ENABLE_VXL}
    -Dfletch_ENABLE_OpenCV:BOOL=${VIAME_ENABLE_OPENCV}
    -DOpenCV_SELECT_VERSION=${VIAME_OPENCV_VERSION}

    -Dfletch_ENABLE_Caffe:BOOL=${VIAME_ENABLE_CAFFE}
    -DAUTO_ENABLE_CAFFE_DEPENDENCY:BOOL=${VIAME_ENABLE_CAFFE}

    -Dfletch_BUILD_WITH_PYTHON:BOOL=${VIAME_ENABLE_PYTHON}

    # Set fletch install path to be viame install path
    -Dfletch_BUILD_INSTALL_PREFIX:PATH=${VIAME_BUILD_INSTALL_PREFIX}

  INSTALL_DIR ${VIAME_BUILD_INSTALL_PREFIX}
  INSTALL_COMMAND ${CMAKE_COMMAND}
    -DVIAME_CMAKE_DIR:PATH=${CMAKE_SOURCE_DIR}/CMake
    -DVIAME_ENABLE_OPENCV:BOOL=${VIAME_ENABLE_OPENCV}
    -DVIAME_BUILD_PREFIX:PATH=${VIAME_BUILD_PREFIX}
    -DVIAME_BUILD_INSTALL_PREFIX:PATH=${VIAME_BUILD_INSTALL_PREFIX}
    -DMSVC=${MSVC}
    -DMSVC_VERSION=${MSVC_VERSION}
    -P ${VIAME_SOURCE_DIR}/CMake/custom_fletch_install.cmake
  )

ExternalProject_Add_Step(fletch forcebuild
  COMMAND ${CMAKE_COMMAND}
    -E remove ${VIAME_BUILD_PREFIX}/src/fletch-stamp/fletch-build
  COMMENT "Removing build stamp file for build update (forcebuild)."
  DEPENDEES configure
  DEPENDERS build
  ALWAYS 1
  )

set( VIAME_ARGS_fletch
  -Dfletch_DIR:PATH=${VIAME_BUILD_PREFIX}/src/fletch-build
  )

set( VIAME_ARGS_Boost
  -DBoost_INCLUDE_DIR:PATH=${VIAME_BUILD_INSTALL_PREFIX}/include
  )

if( VIAME_ENABLE_OPENCV )
  set(VIAME_ARGS_fletch
    ${VIAME_ARGS_fletch}
    -DOpenCV_DIR:PATH=${VIAME_BUILD_PREFIX}/src/fletch-build/build/src/OpenCV-build
    )
endif()

if( VIAME_ENABLE_CAFFE )
  set(VIAME_ARGS_fletch
     ${VIAME_ARGS_fletch}
    -DCaffe_DIR:PATH=${VIAME_BUILD_PREFIX}/src/fletch-build/build/src/Caffe-build
    )
endif()

if( VIAME_ENABLE_VIVIA )
  set(VIAME_ARGS_libkml
     ${VIAME_ARGS_libkml}
    -DKML_DIR:PATH=${VIAME_BUILD_PREFIX}/src/fletch-build/build/src/libkml-build
    )
  set(VIAME_ARGS_VTK
     ${VIAME_ARGS_VTK}
    -DVTK_DIR:PATH=${VIAME_BUILD_PREFIX}/src/fletch-build/build/src/VTK-build
    )
endif()

if( VIAME_ENABLE_VXL )
  set(VIAME_ARGS_VXL
    ${VIAME_ARGS_VXL}
    -DVXL_DIR:PATH=${VIAME_BUILD_PREFIX}/src/fletch-build/build/src/VXL-build
    )
endif()
