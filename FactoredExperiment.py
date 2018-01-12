from itertools import product, repeat
from tempfile import NamedTemporaryFile, TemporaryDirectory
import subprocess
import os
import time
from collections import Iterable
import pdb
import pandas
import copy

import numpy as np
from .utilities import TimedProgressBar, TimedCountProgressBar
from .Experiment import Experiment, get_file_fields, make_fielded_files



class FactoredExperimentIterator:
    '''
    Iterator for FactoredExperiments
    Returns a list of dictionaries
    '''

    def __init__(self, fexpr):
        xfacts = [list(a) for a in [zip(repeat(k),v) for k,v in fexpr]]
        xfacts =  [list(a) for a in product(*xfacts)]
        self._xfacts = xfacts
        self._ndx = 0

    def __len__(self):
        return len(self._xfacts)

    def __getitem__(self, ndx):
        return self._xfacts[ndx]

    def __iter__(self):
        return self

    def __next__(self):
        '''
        I'd rather use a generator here, but then I get stuck in subgenerator
        damnation (PEP380 yield from doesn't seem to solve my problem.)  So,
        let's be old fashioned and use an index, hmm?
        '''
        if self._ndx < len(self._xfacts):
            value = self._xfacts[self._ndx]
            self._ndx = self._ndx + 1
            return value
        raise StopIteration



class FactoredExperiment:
    """
    A factored experiment is one in which configuration parameters are substituted
    from a list.  The cartesian product of all possible values are generated.
    """

    _default_args = ' -s {seed} -set WORLD_X {world_x} -set WORLD_Y {world_y} ' +\
        ' -set DATA_DIR {data_dir} -set ENVIRONMENT_FILE {environment_file} -set EVENT_FILE {events_file}'

    _default_args_dict = {
        'seed':-1,
        'world_x':60,
        'world_y':60,
    }

    _default_events_dict = {
        'interval':100,
        'end':10000
    }

    _default_events_str =\
        'u begin Inject default-heads-norep.org\n' +\
        'u 0:{interval}:end PrintSpatialResources {file_resources}\n' +\
        'u {end} exit'


    def __init__(self, env_str, factors, args_str='', args_dict={}, events_str=None,
        events_dict={}, procs=4, exec_directory='default_config', **kw):
        """
        Initialize the FactoredExperiment.  It is not run until run_experiments()
        is called.

        :param env_str: The environment string to use for the experiment.  Python curly brace {arg}
                            style formatting is used for substitution for factor values.
        :param factors: A list of key,list pairs (or dictionary) that is used to substitute values
                        in the environment string.
        :param args_str: Additional values to append at the end of the default argument string
        :param args_dict:  Additional values or overrides for the default argument string
        :param events_str: Overrides default_events string
        :param events_dict: Addtional values or overrides for the default events string
        :param procs: The number of child subprocesses to spawn at one time
        :param exec_dir: The directory in which to execute the experiments
        """
        self._env_str = env_str
        self._factors = factors
        self._factor_names = [k for k,v in self._factors]
        self._exec_dir = exec_directory
        self._max_procs = procs
        self._reset()
        self._events_str = self._default_events_str if events_str is None else events_str
        self._events_dict = self._default_events_dict
        self._events_dict.update(events_dict)
        self._args_str = self._default_args + ' ' + args_str
        self._args_dict = self._default_args_dict
        self._args_dict.update(args_dict)
        self._used_environment = None
        self._used_events = None
        self._used_args = None

    def _reset(self):
        """
        Purge the object of generated data
        """
        self._ready = False
        self._data = None
        self._data_dir = None
        self._data_dir_handle = None
        self._events_files = []
        self._env_files = []
        self._datafiles = None
        self._stdout_files = []
        self._child_procs = []
        self._child_exit_codes = {}
        self._used_environment = None
        self._used_events = None
        self._used_args = None


    def get_factors(self):
        """
        Return the factors that were used to generate this data.  These are the
        substituted parameters/values.
        """
        return self._factors

    def get_factor_names(self):
        """
        Provide the names of the substituted placeholders
        """
        return self._factor_names

    def run(self, use_pbar=True):
        """
        Actually run the experiments

        :param use_pbar:  If true, use a progress bar

        :returns: self, for chaining
        """

        # If we've already run, reset ourself
        if self._ready:
            self._reset()

        if use_pbar == True:
            pbar = TimedCountProgressBar(title='Running Avida', max_value=len(self))
            pbar.start()
        else:
            pbar = None

        if len(self) > 0:
            # If we have any work to do

            # Create a common data directory for all output
            self._data_dir_handle = TemporaryDirectory()  # the directory will be deleted when this goes out of scope
            self._data_dir = self._data_dir_handle.name

            # Hold on to our active child processes
            active_procs = set()

            self._used_args = self._args_str + '\ndict=' + str(self._args_dict)
            self._used_environment = self._env_str + '\nfactors=' + str(self._factors)
            self._used_events = self._events_str + '\ndict=' + str(self._events_dict)

            try:
                for ndx, settings in enumerate(self):
                    # For each factor combination

                    # Create an output file we can track
                    datafile_paths = make_fielded_files(
                        get_file_fields(self._events_str), self._data_dir)
                    if self._datafiles is None:
                        self._datafiles = {}
                        for field, path in datafile_paths.items():
                            self._datafiles[field] = [path]
                    else:
                        for field, path in datafile_paths.items():
                            self._datafiles[field].append(path)

                    # Create our experiment-specific events file (for the data filename differs)
                    with NamedTemporaryFile(dir=self._data_dir, mode='w', delete=False) as events_file:
                        events_str = self._events_str\
                            .format(**{**datafile_paths, **self._events_dict})
                        events_file.write(events_str)
                        self._events_files.append(events_file.name)

                    # Patch our default environment dictionary with our experiment
                    # factors' settings and create our environment file
                    env_str = self._env_str.format(**dict(settings))
                    with NamedTemporaryFile(dir=self._data_dir, mode='w', delete=False) as env_file:
                        env_file.write(env_str)
                        self._env_files.append(env_file.name)

                    # Set up our arguments
                    args_str = self._args_str.format(
                        events_file=self._events_files[-1],
                        environment_file=self._env_files[-1],
                        data_dir = self._data_dir,
                        **self._args_dict)

                    # Create an output file to hold our stdout/err streams
                    self._stdout_files.append(NamedTemporaryFile(dir=self._data_dir, delete=False, mode='w'))


                    # Launch a chidl process
                    child_proc = subprocess.Popen('./avida ' + args_str,
                                    cwd=self._exec_dir,
                                    shell=True,
                                    stdout=self._stdout_files[-1],
                                    stderr=subprocess.STDOUT,
                                    universal_newlines=True)
                    self._child_procs.append(child_proc)
                    active_procs.add(child_proc)

                    # Detect a full pool and wait for a spot to open
                    if len(active_procs) >= self._max_procs:
                        self._update_pbar(pbar)
                        time.sleep(.1)
                        active_procs.difference_update(
                            [p for p in active_procs if p.poll() is not None])

            except StopIteration:
                pass

            # Finish up
            while len(active_procs) > 0:
                self._update_pbar(pbar)
                time.sleep(.1)
                if pbar:
                    self._update_pbar(pbar)
                active_procs.difference_update(
                    [p for p in active_procs if p.poll() is not None])


            self._update_pbar(pbar)

            # If there were non-zero exit codes, dump the output
            # and make a note that things went wrong
            was_errors = False
            for ndx, p in enumerate(self._child_procs):
                if p.returncode != 0:
                    self._dump_error(ndx)
                    was_errors = True


        if pbar:
            pbar.finish()

        # Don't report ready if there were errors
        self._ready = not was_errors

        # Return ourself for chaining if there were no errors, otherwise return None
        self._load_data()
        return self if not was_errors else None


    def results(self):
        """
        Return the data we've generated as pairs of the factors we substituted and a
        Pandas DataFrame.

        :return: If the data is ready, return it otherwise raise an error
        """
        if not self._ready:
            raise LookupError('The experiments have not been completed.')
        else:
            return self._data

    def keys(self):
        if not self._ready:
            raise LookupError('The eperiments have not been completed.')
        else:
            return self._data.keys()

    def config(self):
        config = {}
        config['args'] = self._used_args
        config['events'] = self._used_events
        config['environment'] = self._used_environment
        return config

    def get_paths(self):
        return self._datafile_paths


    def _load_data(self):
        if self._data == None:
            self._data = {}
            for field in self._datafiles.keys():
                self._data[field[5:]] = []
            for ndx, settings in enumerate(self):
                for field, paths in self._datafiles.items():
                    d = pandas.read_csv(paths[ndx], comment='#',
                            skip_blank_lines=True, delimiter=' ', header=None)
                    # Strip file_ from our field to create a key
                    self._data[field[5:]].append( (settings, d) )


    def animate(self, data_transform=None, figkw={}, animkw={}):
        """
        A helper method to animate the resources

        :param data_transform: A function to transform our Pandas DataFrame
        :param figkw: KW arguments to pass to the animation object's initializer
        :param animkw: KW arguments to pass to the animation object's animation method

        :return: the animation object.  Not this has to be converted to html5_video
                 or saved before the rendering will actually occur.
        """
        if data_transform is not None:  # Transform the data if requested
            self._data['resources'] = data_transform(self._data['resources'])
        return FactoredCategoricalHeatmapAnimation(
            self._data['resources'], self.world_size(), self.get_dims(), **figkw).animate(**animkw)



    def world_size(self):
        """
        :return: a tuple of the world size
        """
        return self._args_dict['world_x'], self._args_dict['world_y']


    def dims(self):
        """
        Return the dimensions of our factored experiment.
        """
        dims = [len(val) for key,val in self._factors]
        return dims

    def _update_pbar(self, pbar):
        """
        Helper function to animate the progress bar during the course of child process
        execution.
        """
        if pbar:
            pbar.update(sum(map(lambda x: 1 if x.returncode is not None else 0, self._child_procs)))
        return


    def _dump_error(self, ndx):
        """
        Dump the stdout/err for a process and its exit code.

        :param ndx:  The experiment's index
        """
        print('For Settings {}'.format(self[ndx]))
        print ('EXIT CODE: {}'.format(self._child_procs[ndx].returncode))
        print('ERROR RUNNING EXPERIMENT.  STDOUT/ERR FOLLOWS')
        print('---------------------------------------------')
        with open(self._stdout_files[ndx].name, 'r') as f:
            print(f.read())
        print('\n\n\n')


    def __getitem__(self, key):
        """
        Return the factors and data given a particular index
        """
        return self._data[key]

    def __iter__(self):
        """
        Return an iterator over all the experiments
        """
        return FactoredExperimentIterator(self._factors)


    def __len__(self):
        return len(self.__iter__())
