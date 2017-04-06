
from viame.processes.benthos_detect import ProcessImage_sri

def __sprokit_register__():
    from sprokit.pipeline import process_factory

    module_name = 'python:benthos_detect.ProcessImage_sri'

    if process_factory.is_process_module_loaded( module_name ):
        return

    process_factory.add_process('ProcessImage_sri', 'SRI Benthos Detector', ProcessImage_sri.ProcessImage_sri)

    process_factory.mark_process_module_as_loaded( module_name )
