source /home/david/zdata/projects/VIAME/build/install/setup_viame.sh
export PYTHONPATH=/home/david/zdata/projects/VIAME/packages/kwiver/sprokit/processes/python:$PYTHONPATH
export PYTHONPATH=/home/david/zdata/projects/VIAME/build/install/lib/python2.7/dist-packages:$PYTHONPATH

export PYTHONPATH=/home/david/zdata/projects/VIAME/build/install/python:$PYTHONPATH
export PYTHONPATH=/home/david/zdata/projects/fishtrack/python:$PYTHONPATH

export PYTHONPATH=/home/david/zdata/projects/fishtrack/local/lib/python2.7:$PYTHONPATH
export PYTHONPATH=/home/david/zdata/projects/fishtrack/local/lib/python2.7/dist-packages:$PYTHONPATH
export PYTHONPATH=/home/david/zdata/projects/fishtrack/PythonModule/build:$PYTHONPATH
export PYTHONPATH=/home/david/zdata/projects/fishtrack/python:$PYTHONPATH

export LD_LIBRARY_PATH=/home/david/zdata/projects/fishtrack/local/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/david/zdata/projects/fishtrack/local/lib64:$LD_LIBRARY_PATH

/home/david/zdata/projects/VIAME/build/install/bin/pipeline_runner -S pythread_per_process -p /home/david/zdata/projects/VIAME/examples/BenthosDetect/images_to_python.pipe 
