import tempfile
import string
import pdb


class Experiment:

    def __init__(self):
        pass

    def run(self, **kw):
        raise NotImplemented

    def results(self):
        """
        Return an iterable of keys and either a PandasDataFrame or
        iterable of PandasDataFrames
        """
        raise NotImplemented

    def keys(self):
        """
        Return the names for the different results
        """
        raise NotImplemented

    def config(self):
        """
        Return a dictionary of configuration settings
        """
        raise NotImplemented

    def get_paths(self):
        """
        Return a dictionary of data files keyed to their configuration
        file field name with value(s) of data files.
        """
        raise NotImplemented

    def __getitem__(self, key):
        raise NotImplemented


def get_file_fields(s):
    file_fields = []
    stripped_fields = []
    parsed = string.Formatter().parse(s)
    for literal, field, fspec, conv in parsed:
        if field is not None and field.startswith('file_'):
            file_fields.append(field)
            stripped_fields.append(field[5:])
    return file_fields


def make_fielded_files(fields, path):
    files = {}
    for field in fields:
        with tempfile.NamedTemporaryFile(dir=path, mode='w', delete=False) as tmpfile:
            files[field] = tmpfile.name
    return files
