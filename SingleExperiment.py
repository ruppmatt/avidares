import pdb
import pandas
import subprocess
import tempfile
import progressbar
import time
import types

import numpy as np
import seaborn as sb
import matplotlib as mpl
from matplotlib import animation, rc
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.text import Text
import pdb
from collections import Iterable

from IPython.display import HTML

from .utilities import TimedSpinProgessBar, TimedProgressBar, write_temp_file,\
    ColorMaps, TitleElapsedProgressBar, blend, check_mask

from .Experiment import Experiment, get_file_fields, make_fielded_files




class SingleExperiment(Experiment):

    default_args = '-s -1'
    default_events ='\
    u begin Inject default-heads-norep.org\n\
    u 0:100:end PrintSpatialResources {file_resources}\n\
    u 25000 exit\n'


    def __init__(self, environment, world_size, cwd='default_config', args=None, events=None, use_pbar=True):
        """
            :param environment:  A string representation of the environment file.  Required.
            :param world_size:   A tuple of the (X,Y) size of the world.  Required.
            :param cwd:  The working directory to execute Avida.  Optional.
            :param args:  Arguments to pass to Avida aside from world size and location of input/output files.  Optional.
            :param evnets: The contents of the events file.  If not provided, a default is used. Optional
            :param use_pbar: Show the progress bar
        """
        Experiment.__init__(self)
        self._cwd = cwd
        self._world_size = world_size
        self._args = args if args is not None else self.default_args
        self._environment = environment
        self._events = events if events is not None else self.default_events
        self._pbar = TimedSpinProgessBar('Running experiment') if use_pbar else None
        self._data = None
        self._datafile_paths = None
        self._used_events = None
        self._used_environment = None
        self._used_args = None


    def run(self):
        """
        Actually run the experiment and load the results.

        :return: self for the purpose of chaining
        """
        args = self._args
        args += f' -set WORLD_X {self._world_size[0]} -set WORLD_Y {self._world_size[1]}'

        # Create a temporary directory to hold our avida output
        self._data_dir = tempfile.TemporaryDirectory()
        args += f' -set DATA_DIR {self._data_dir.name}'

        # Write our environment file
        path = write_temp_file(self._environment, dir=self._data_dir.name)
        args += f' -set ENVIRONMENT_FILE {path}'

        # Work with the events file
        # Fill in our file paths
        self._datafile_paths = make_fielded_files(
            get_file_fields(self._events), self._data_dir.name)

        # Write out our events
        self._events = self._events.format(**self._datafile_paths)
        path = write_temp_file(self._events, dir=self._data_dir.name)
        args += f' -set EVENT_FILE {path}'

        # Store for later use
        self._used_args = args
        self._used_environment = self._environment
        self._used_events = self._events

        # Run avida
        self._run_process('./avida ' + args)

        # Load our data files
        self._data = {}
        for field, path in self._datafile_paths.items():
            # Strip "file_" off the field name and use it as a key
            # And load the file in as a Pandas DataFrame
            self._data[field[5:]] = pandas.read_csv(path,
                                    comment='#', skip_blank_lines=True,
                                    delimiter=' ', header=None)
        return self


    def results(self):
        return self._data

    def keys(self):
        return self._data.keys()

    def datafile_paths(self):
        return self._datafile_paths

    def config(self):
        config = {}
        config['args'] = self._used_args
        config['events'] = self._used_events
        config['environment'] = self._used_environment
        return config

    def world_size(self):
        return self._world_size

    def __getitem__(self, key):
        return self._data[key]


    def animate(self, data_transform=None, figkw={}, animkw={}):
        """
        A helper method to animate using ResourceExperimentAnimation.

        :param data_transform: A function to transform our Pandas DataFrame
        :param figkw: KW arguments to pass to the animation object's initializer
        :param animkw: KW arguments to pass to the animation object's animation method

        :return: the animation object.  Not this has to be converted to html5_video
                 or saved before the rendering will actually occur.
        """
        # Generate our data
        if data_transform is not None:  # Transform the data if requested
            self._data = data_transform(self._data)

        return ResourceExperimentAnimation(self._data, world_size=self._world_size, **figkw).animate(**animkw)


    def _run_process(self, args):
        """
        An internal helper function to actually run the subprocess.
        :param args: The commandline argument to execute
        """


        # subprocess.PIPE can only hold about 64k worth of data before
        # it hangs the chid subprocess.  To get around this, we're writing
        # the standard output and standard error to this temporary file.
        tmp_stdout = tempfile.NamedTemporaryFile(mode='w', delete=False)
        file_stdout = open(tmp_stdout.name, 'w')

        # Spawn the child subprocess.  We're using Popen so we can animate
        # our spinning progressbar widget.  We're using the shell=True to
        # deal with issues trying to spawn avida properly with the command
        # line options constructed as a single string.
        # This may not work properly on Windows because reasons.  There's
        # a lot of dicussion online about how to alter it so that it does
        # work in Windows.

        proc = subprocess.Popen(args,
                                cwd=self._cwd,
                                shell=True,
                                stdout=file_stdout,
                                stderr=subprocess.STDOUT,
                                universal_newlines=True)
        if self._pbar:
            self._pbar.start()

        # Wait for our process to finish; poll() will return the exit
        # code when it's done or None if it is still running.  The wheel
        # spins via the update().
        while proc.poll() is None:
            time.sleep(0.25)
            if self._pbar:
                self._pbar.update()
        return_code = proc.wait()  # Grab our subprocess return code
        file_stdout.close()  # Close our subprocess's output streams file

        if self._pbar:
            self._pbar.finish(return_code)  # Finish the progress bar

        # Handle issues if the process failed.
        # Print the standard output / error from the process temporary
        # file out, then raise a CalledProcessError exception.
        if return_code != 0:
            with open(tmp_stdout.name) as file_stdout:
                print(file_stdout.read())
            raise subprocess.CalledProcessError(return_code, args)
