import importlib.util


class ImportConfigUtil:

    def import_module(self, file_path):
        module_name = "config"
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
