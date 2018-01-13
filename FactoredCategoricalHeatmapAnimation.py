from collections import Iterable
import pdb
import pandas
import copy
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
import seaborn as sb
import numpy as np
import collections

from .utilities import ColorMaps, TimedProgressBar, TimedCountProgressBar, blend, check_mask



class FactoredCategoricalHeatmapAnimation:
    """
    Draw an experiment that can have multiple factors.  We're limited to two dimensions, but
    not particularly by the number of factors in each of those two dmensions.
    """

    _multi_cat_cmap = [ColorMaps.green, ColorMaps.red, ColorMaps.blue]

    def __init__(self, heatmap_data, grid_shape, dims=(1,),
                 title='', cmap=None, use_pbar=True, interval=50,
                 post_plot=[], env_str='', event_str='', lower_mask_thresh=0.0, **kw):
        self._ndx_t = 0
        self._ndx_cat = 1
        self._ndx_cell = 2
        self._heatmap_data = copy.copy(heatmap_data)
        self._dims = dims
        self._factors = None
        self._categories = None    #Name of resources
        self._is_multi = None     #Are we plotting multiple categories?
        self._grid_shape = grid_shape  #The size of the Avida world
        self._num_frames = None   #How many frames are we drawing?
        self._interval = interval  #How fast should the animation go?
        self._to_draw = None    #With blitting, what artists do we need to draw?
        self._post_plot = post_plot     #After the axes is drawn, what else should we draw on it?

        self._fig = None        #A handle to our figure
        self._last_anim = None     # A cached copy of the last animation rendered; force=True in animat() will replace it.

        self._vmin = None   # Maximum value in our dataset
        self._vmax = None   # Minimum value in our dataset
        self._prepare_data()    # Now let's prepare our data

        self._cmap = ColorMaps.green if cmap is None else cmap  #What colormap(s) are we using?
        self._colors = ['green', 'red', 'blue']  #If multi, how should the legend patches be colored

        self._pbar =\
            TimedProgressBar(title='Building Animation', max_value=self._num_frames) if use_pbar else None
        self._title = title  # The title of the plot
        self._env_str = env_str
        self._event_str = event_str
        if not self._is_multi:  #Handle our colormaps
            self._cmap = ColorMaps.green if cmap is None else cmap
        else:
            self._cmap = self._multi_cat_cmap if cmap is None else cmap
        self._lower_mask_thresh = lower_mask_thresh


    def _prepare_data(self):
        """
        Internal function to setup our animation.  We gather data about the
        number of resources, how many frames we're going to draw, and what our
        min and max values are, and how our factors are set up
        """
        self._factors = []
        if isinstance(self._heatmap_data, pandas.DataFrame):
            # Make it look multifactored if it is a single experiment
            self._heatmap_data = [ ( [('',''),('','')] , self._heatmap_data) ]
        for facts,data in self._heatmap_data:
            if self._num_frames is None:
                self._num_frames = len(data.iloc[:,self._ndx_t].unique())
            if self._categories is None:
                self._categories = np.unique(data.iloc[:,self._ndx_cat])
                if len(self._categories) > 1:
                    self._is_multi = True
                if len(self._categories) > 3:
                    raise ValueError('Animations are currently limited to three categories')
            self._factors.append(facts)
            abundances = data.iloc[:,self._ndx_cell:].astype('float')
            d_min = abundances.min().min()
            d_max = abundances.max().max()
            if self._vmin is None or d_min < self._vmin:
                self._vmin = d_min
            if self._vmax is None or d_max > self._vmax:
                self._vmax = d_max



    def setup_figure(self):
        """
        A helper class for our init_func used during the animation process.
        This class sets up all objects that will be animated assuming blitting
        is enabled.  (Blitting speeds up animation *considerably*.)
        """
        # How many data plots are we dealing with in each dimension?
        plots_x = self._dims[0]  # Number of columns
        plots_y = self._dims[1]  if len(self._dims) > 1 else 1 # Number of rows

        # Set up our base row count
        num_rows = plots_y + 1  # Add one more row for the update number
        height_ratios = [1] * plots_y + [0.25]
        num_cols = plots_x + 1  # Add one more column for the colorbar
        width_ratios = [1] * plots_x + [0.10]

        if self._is_multi:
            # If we have multiple resources, add another row for the resource legend
            num_rows += 1
            height_ratios.append(0.1)

        has_descr = True if len(self._env_str + self._event_str) > 0 else False
        if has_descr:
            # if we need to print some descriptive text, add another at the bottom
            # change this height ratio to make it larger
            num_rows += 1
            height_ratios.append(0.35)

        # Create our grid layout
        gs = mpl.gridspec.GridSpec(num_rows, num_cols,
                           height_ratios=height_ratios,

                           width_ratios=width_ratios)

        # Plot our category heatmaps
        ndx = 0  # Index into our experiment
        plots = []  # Plots from our experiment
        for col in range(plots_x):
            for row in range(plots_y):
                ax = plt.subplot(gs[row,col])
                base_cmap = self._cmap if not self._is_multi else ColorMaps.gray
                plot = plt.imshow(np.zeros(self._grid_shape), cmap=base_cmap,
                    origin='upper', interpolation='nearest',
                    vmin=self._vmin, vmax=self._vmax)
                ax.tick_params(axis='both', bottom='off', labelbottom='off',
                               left='off', labelleft='off')
                if self._is_left_edge(ndx):
                    ax.set_ylabel(self._fact2label(ndx,1))
                if self._is_bottom_edge(ndx):
                    ax.set_xlabel(self._fact2label(ndx,0))
                plots.append(plot)
                pa = []
                for pp in self._post_plot:
                    pa.append(pp.blit_build(ax, ax_ndx=ndx))
                ndx = ndx+1

        # Plot the colorbar
        norm = mpl.colors.Normalize(self._vmin, self._vmax)
        cax = plt.subplot( gs[0:plots_y,-1] )  # Across data rows, last column
        if not self._is_multi:
            cbar = mpl.colorbar.ColorbarBase(cax, cmap=self._cmap, norm=norm, orientation='vertical')
        else:
            cbar = mpl.colorbar.ColorbarBase(cax, cmap=ColorMaps.gray, norm=norm, orientation='vertical')
        cbar.set_label('Abundance')

        # Plot the update
        ax = plt.subplot(gs[plots_y,0:plots_x])  # The row after the data plots, across all data plot columns
        ax.tick_params(axis='both', bottom='off', labelbottom='off',
                       left='off', labelleft='off')
        ax.set_frame_on(False)
        ax.set_ylim(0,1)
        ax.set_xlim(0,1)
        update = ax.text(0.5,0.25,'Update n/a', ha='center', va='bottom')

        # Plot the category legend if needed
        if self._is_multi:
            ax = plt.subplot(gs[plots_y+1,:-1])  # The row after the update axis, acros all data plot columns
            legend_handles = []
            for ndx,cat_name in enumerate(self._categories):
                legend_handles.append(mpl.patches.Patch(color=self._colors[ndx], label=cat_name))
            plt.legend(handles=legend_handles, loc='center', frameon=False, ncol=len(legend_handles))
            ax.tick_params(axis='both', bottom='off', labelbottom='off',
                                   left='off', labelleft='off')
            ax.set_frame_on(False)

        # If we have an environment and event strings, plot them in the final row across all columns
        if has_descr:
            ax = plt.subplot(gs[-1,:])
            desc = self._env_str + '\n\n' + self._event_str + '\n\n' + f'World: {self._world_size[0]} x {self._world_size[1]}'
            env = ax.text(0.05, 1, desc, ha='left', va='top', fontsize=7)
            ax.tick_params(axis='both', bottom='off', labelbottom='off',
                                   left='off', labelleft='off')
            ax.set_frame_on(False)

        # Title the figure
        plt.suptitle(self._title)

        # Store what we need to redraw each frame for blitting.
        # The values in this dictionary may be either a single element
        # or an iterable.
        self._to_draw = {'plots':plots, 'update':update, 'post_plot':pa}


    def get_drawables(self):
        """
        A helper function to get all artist objects that should be redrawn
        each frame during blitting.  Any values that are iterables in the
        dictionary are flattened into the list that is returned.

        :return: List of artist to be drawn each frame
        """
        to_draw = []
        for k,v in self._to_draw.items():
            if isinstance(v,Iterable):
                for i in v:
                    to_draw.append(i)
            else:
                to_draw.append(v)
        return to_draw


    def _is_left_edge(self, ndx):
        """
        Given an experiment's index, is it on the left edge of the plot?

        :param ndx: Experiment index

        :return: True if on the left edge; False otherwise
        """
        if len(self._dims)== 1:
            return ndx == 0
        return ndx < self._dims[1]


    def _is_bottom_edge(self, ndx):
        """
        Give an experiment's index, is it on the bottom edge of the plot?

        :param ndx: Exepriment index

        :return: True if on the bottom edge, False otherwise
        """
        if len(self._dims) == 1:
            return True
        return (ndx % self._dims[1]) == self._dims[1]-1


    def _fact2label(self, ax_ndx, fact_ndx):
        """
        Return the axis label for an experiment.

        :param ax_ndx: the experiment's plot index
        :param fact_ndx: the factor order (e.g. 0th or 1st factor)

        :return: the string label for the axis
        """
        if len(self._dims) > 1:
            key,value = self._factors[ax_ndx][fact_ndx]
        else:
            if fact_ndx == 1:
                return ''
            key,value = self._factors[ax_ndx][0]
        return '{} = {}'.format(key,value) if key != '' else ''


    def __getitem__(self, key):
        """
        A helper utility to return the artists associated with a particular
        key in the _to_draw dictionary.

        :param: Name of artist to return

        :return: An artist
        """
        return self._to_draw[key]


    # ===========================================
    # What follows are the three classes that are needed to create animation
    # plot with blitting: InitFrame, which sets up the figure before the
    # first frame is drawn.  This is important for blitting.  GenerateFrameData,
    # which is used to inform the frame drawer about any information it needs
    # to draw the frame.  Finally, DrawFrame, which actually does the drawing.
    #
    # I made these classes because I wanted to pass common information from
    # the ResourceExperimentAnimation class to them, and I needed a way to
    # keep within the signature restrictions placed on them as components of the
    # core animation function, FuncAnimation.
    # ===========================================

    class InitFrame:
        """
        Before the first frame is drawn, setup the figure.  For blitting,
        all objects that change over the course of the animation need to be
        created and returned.
        """
        def __init__(self, setup):
            setup.setup_figure()
            self._setup = setup

        def __call__(self):
            return self._setup.get_drawables()


    class GenerateFrameData:
        """
        A generator that yields information necessary for each frame.  It
        serves as the argument for FuncAnimation's frames parameter.

        Note that the number of times this iterator is called depends on
        the value of FuncAnimation's save_count.
        """
        def __init__(self, setup):
            self._setup = setup

        def __call__(self):
            """
            The generator function itself.  It returns the data needed to
            alter the artists in each frame of the animation

            :return: A tuple of objects needed by the animation method.  In
                     This case, that's DrawFrame's __call__ method's first
                     positional parameter.
            """

            if self._setup._pbar is not None:
                self._setup._pbar.start()

            heatmap_data = self._setup._heatmap_data  # Just to make our lives easier, give it a name
            updates = np.unique(heatmap_data[0][1].iloc[:,self._setup._ndx_t])  # Get the list of updates we will be working with
            grid_x, grid_y = self._setup._grid_shape

            if self._setup._is_multi:
                # Multiple-category experiments need to be blended, so they are handled differently
                blended = []  # Will hold the blended values for each update by experiment, then by cell and by color channel
                num_categories = len(self._setup._categories)

                # Grab the colorbars we're going to use for the blending
                colors = list(map(lambda x: x.colors, self._setup._cmap[0:num_categories]))

                for factors, expr_data in heatmap_data:
                    # Blend each experiment
                    blended.append(blend(expr_data, colors, self._setup._categories))

                for u_ndx, update in enumerate(updates):
                    # Enumerate each update and plot the proper experiment
                    data = []
                    mask = []
                    for e_ndx, bdata in enumerate(blended):
                        data.append(bdata[u_ndx].reshape(grid_x, grid_y, 3))
                        expr_data = heatmap_data[e_ndx][1]
                        #pdb.set_trace()
                        update_data = \
                            expr_data[expr_data.iloc[:,self._setup._ndx_t]==update]\
                            .iloc[:,self._setup._ndx_cell:]
                        sum_update_data = update_data.sum(axis=0)
                        mask.append(
                            np.ma.masked_less_equal(
                                sum_update_data, self._setup._lower_mask_thresh)\
                            .reshape(grid_x, grid_y))
                    yield u_ndx, update, data, mask

            else:
                # We're not doing blending, just iterate through the data we have
                for ndx, update in enumerate(updates):
                    data = []
                    mask = []
                    update = heatmap_data[0][1].iloc[ndx,self._setup._ndx_t]
                    for factors, expr_data in heatmap_data:
                        data.append(\
                            expr_data.iloc[ndx,self._setup._ndx_cell:]\
                            .astype('float')\
                            .values.reshape(self._setup._grid_shape))
                        update_data = \
                            expr_data[expr_data.iloc[:,self._setup._ndx_t]==update]\
                            .iloc[:,self._setup._ndx_cell:]
                        sum_update_data = update_data.sum(axis=0)
                        mask.append(
                            np.ma.masked_less_equal(
                                sum_update_data, self._setup._lower_mask_thresh)\
                            .reshape(self._setup._grid_shape))
                    yield ndx, update, data, mask

            raise StopIteration


    class DrawFrame:
        """
        This is the class that actually draws each frame.  It is the first
        required parameter of FuncAnimation.  This class's __call__ signature
        matches the requirements established by FuncAnimation.
        """

        def __init__(self, setup):
            self._setup = setup

        def __call__(self, info, *fargs):
            """
            This is the method that alters the figure for each frame.

            :param info: A tuple from the frame generator (DataFrameGenerator, here)
            :param fargs: A list of arguments passed via FuncAnimation's fargs parameter

            :return: An iterable of artists to draw
            """
            frame = info[0]  # Frame number
            update = info[1] # Update value
            grid_data = info[2]  # Data to draw our grids
            mask = info[3] # Mask of data
            self._setup['update'].set_text(f'Update {update}')
            for ndx,data in enumerate(grid_data):
                self._setup['plots'][ndx].set_array(check_mask(data,mask[ndx]))
                for pp in self._setup['post_plot']:
                    pp.blit_update(frame, update, ax_ndx=ndx)
            if self._setup._pbar:
                self._setup._pbar.update(frame)
                if frame == self._setup._num_frames - 1:
                    self._setup._pbar.finish()
            return self._setup.get_drawables()



    def animate(self, force=False, blit=True, **kw):
        """
        Setup the animation request using FuncAnimation.  Note that this method
        does *not* actually perform the animation until it is either displayed
        (by to_html5_video or the like) or saved.  Until then a handle to the
        what is returned by FuncAnimation must be held otherwise the garbage
        collector will eat it.

        :param force: Do not use a cached copy of the animation object
        :param blit: Should we use blitting to speed animation *considerably*?

        :return: A handle to the animation
        """
        if self._last_anim is not None and force == False:
            return self._last_anim

        # We need to create the figure before we call FuncAnimation as it is
        # a required argument.
        if 'fig_conf' in kw:
            self._fig = plt.figure(**kw['fig_conf'])
        else:
            self._fig = plt.figure()


        # We're initializing these helper classes with ourself because we want
        # all the setup information maintained.  The __call__ methods for these
        # nested classes match the signature expected by FuncAnimation for their
        # various purposes

        # Helper that initializes the figure before the first frame is drawn
        init_fn = FactoredCategoricalHeatmapAnimation.InitFrame(self)

        # Helper that generates the data that is used to adjust each frame
        frame_gen = FactoredCategoricalHeatmapAnimation.GenerateFrameData(self)

        # Helper that updates the contents of the figure for each frame
        frame_draw = FactoredCategoricalHeatmapAnimation.DrawFrame(self)

        # The actual animation creation call itself.  A handle to the return
        # value must be kept until the animation is rendered or the garbage
        # collector wille eat it
        anim = animation.FuncAnimation(self._fig,
                                   frame_draw,
                                   init_func=init_fn,
                                   frames=frame_gen,
                                   fargs=[], interval=self._interval, save_count=self._num_frames,
                                   blit=blit)
        self._last_anim = anim
        self._fig.show(False)   # Try to hide the figure; it probably won't
        return anim
