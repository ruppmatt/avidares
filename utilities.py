import progressbar
import seaborn as sb
from tempfile import NamedTemporaryFile, TemporaryFile, TemporaryDirectory
from functools import reduce
import multiprocessing as mproc
import numpy as np
import matplotlib as mpl
import pdb
import pickle
import subprocess
import os
import pyximport; pyximport.install(
    setup_args={
        'include_dirs':np.get_include()
        })
from .blender import blender


def save_n_html(path, anim, config, animkw={}):
    a = anim.animate(animkw=animkw)
    html = a.to_html5_video()
    title = anim._title.replace('\n', ' ')
    title = title.replace('/', '-')
    title = title.replace(':', '-')
    comments = ''
    for field, values in config.items():
        comments += '[{}]'.format(field.upper())
        comments += values + '\n\n'
    metadata = {
        'artist':'AvidaED Development',
        'title':f'{anim._title}',
        'description':comments
    }
    path = f'{path}/{title}.mp4'
    a.save(path, dpi=300, metadata=metadata)
    mpl.pyplot.close()
    return html

def check_mask(data, mask):
    """
    Because of an issue with imshow and fully masked data,
    we're going to apply this work-around that will give us
    a white image if there is no data to display.

    Otherwise, given data, return a masked dataset, broadcasting to
    the color dimensions if needed
    """
    if mask.count() == 0:
        ws = data.shape
        empty = np.matlib.repmat(np.ones(3), ws[0], ws[1])
        return empty.reshape(ws[0], ws[1], -1)
    else:
        if data.ndim == 3:
            _mask = np.ma.getmaskarray(mask)[:,:,np.newaxis]
            dmask = np.broadcast_to(_mask, data.shape)
            masked = np.ma.array(data=data, mask=dmask)
        else:
            masked = np.ma.array(data=data, mask=np.ma.getmaskarray(mask))
        return masked

def blend(data, cmaps, res_names):
    """
    Call out to our blender to blend the colors in multi-resource experiments.

    :param data: either a Pandas DataFrame or list of them
    :param cmaps: the color maps we're associating with each resource abundance
    :param res_names: the names of the resources

    :return: A numpy floating point array Updates x Num_Cells x 3 (RGB) or a
             list of them.
    """

    # Our blender will only take floating points values to improve
    # efficiency.  So, we have to convert each res_name to a floating point.
    res_num = {k:n for n,k in enumerate(res_names)}


    if isinstance(data, list):  #If we're dealing with a list
        retval = []
        for d in data:
            _data = d.copy()  # Let's not muck with the original
            for k in range(_data.shape[0]):
                _data.iloc[k,1] = res_num[data.iloc[k,1]]  # Convert resource names to numbers
            retval.append(blender(np.array(_data, dtype=float), np.array(cmaps, dtype=float)))
        return retval
    else:  # If we're dealing with a single data frame
        _data = data.copy()  # Let's not muck with the original
        for k in range(_data.shape[0]):
            _data.iloc[k,1] = res_num[data.iloc[k,1]] # Convert resource names to numbers
        return blender(np.array(_data, dtype=float), np.array(cmaps, dtype=float))




def cubehelix_palette(n_colors=6, start=0, rot=.4, gamma=1.0, hue=0.8,
                      light=.85, dark=.15, reverse=False, as_cmap=False):
    """
        I'm going to monkey patch the original cubehelix_palette function in
        Seaborn to make it return a matplotlib colormap with the same number of
        colors as requested by the function definition.  In the original source
        code, 256 values are returned.  This is too many.
    """
    cdict = mpl._cm.cubehelix(gamma, start, rot, hue)
    cmap = mpl.colors.LinearSegmentedColormap("cubehelix", cdict)

    x = np.linspace(light, dark, n_colors)
    pal = cmap(x)[:, :3].tolist()
    if reverse:
        pal = pal[::-1]

    if as_cmap:
        x_nc = np.linspace(light, dark, n_colors)
        if reverse:
            x_nc = x_nc[::-1]
        pal_nc = cmap(x_nc)
        cmap = mpl.colors.ListedColormap(pal_nc)
        return cmap
    else:
        return sb.palettes._ColorPalette(pal)

# Perform monkey-patching
sb.cubehelix_palette = cubehelix_palette



def write_temp_file(contents, **kw):
    """
    Write contents to a temporary file that is *not* deleted when the
    handle to the file is lost.  I use a TemporaryDirectory to hold these
    files so they do eventually get cleaned up in this code.  The OS will also
    clean them up on a restart.

    :param contents: what to write to the temporary file

    :return: the path to the temporary file.
    """
    fn = NamedTemporaryFile(mode='w', delete=False, **kw)
    fn.write(contents)
    path = fn.name
    fn.close()
    return path



class UpdateIterator:
    """
    Because each update might have mutliple resources associated with it, we
    sometime want to iterate through our data either one or multiple updates
    at a time.  This iterator provides that functionality for our expected
    Pandas DataFrame.
    """

    def __init__(self, data, chunk_size=1):
        """
        :param data: the Pandas DataFrame containing our resource data
        :param chunk: the number of updates to return per iteration
        """
        self._data = data
        self._chunk_size = chunk_size

    def __iter__(self):
        self._ndx=0
        return self

    def __next__(self):
        updates = self._data.iloc[:,0].unique()
        while self._ndx < len(updates):
            chunk_updates = updates[self._ndx:self._ndx+self._chunk_size]
            chunk_ndxs = None
            for update in chunk_updates:
                update_rows = self._data.iloc[:,0] == update
                chunk_ndxs = update_rows if chunk_ndxs is None else np.logical_or(chunk_ndxs, update_rows)
            self._ndx += self._chunk_size
            return self._data[chunk_ndxs]
        raise StopIteration()




class ColorMaps:
    """
    Just a class to store some common colormaps
    """
    green = sb.cubehelix_palette(
        start=2, rot=0, hue=1, dark=0.10, light=0.90, gamma=0.7, n_colors=16,
        as_cmap=True)
    blue = sb.cubehelix_palette(
        start=0, rot=0, hue=1, dark=0.10, light=0.90, gamma=0.7, n_colors=16,
        as_cmap=True)
    red = sb.cubehelix_palette(
        start=1, rot=0, hue=1, dark=0.10, light=0.90, gamma=0.7, n_colors=16,
        as_cmap=True)
    gray = sb.cubehelix_palette(
        start=0, rot=0, hue=0, dark=0.10, light=0.90, gamma=0.7, n_colors=16,
        as_cmap=True)




class TimedProgressBar(progressbar.ProgressBar):
    """
    A progress bar that fills as progress is made, displays a percentage
    completed, estimated time remaining, and time done when finished.
    """
    def __init__(self, title='', **kw):

        self._pbar_widgets = [\
                        title,
                        '  ',
                        progressbar.Bar(),
                        progressbar.Percentage(),
                        '  ',
                        progressbar.ETA(
                            format_zero='%(elapsed)s elapsed',
                            format_not_started='',
                            format_finished='%(elapsed)s elapsed',
                            format_NA='',
                            format='%(eta)s remaining')]
        progressbar.ProgressBar.__init__(self, widgets=self._pbar_widgets, **kw)




class TimedCountProgressBar(progressbar.ProgressBar):
    """
    A progressbar that tells how many of the total objects are done,
    the estimated time remaining, and the total elapsed time when finished.
    """
    def __init__(self, title='', **kw):
        self._pbar_widgets = [
            title,
            ' ',
            progressbar.FormatLabel('(%(value)s of %(max_value)s)'),
            ' ',
            progressbar.ETA(
                format_zero='%(elapsed)s elapsed',
                format_not_started='',
                format_finished='%(elapsed)s elapsed',
                format_NA='',
                format=''
                )
            ]
        progressbar.ProgressBar.__init__(self, widgets=self._pbar_widgets, **kw)



class TimedSpinProgessBar(progressbar.ProgressBar):
    """
    A progressbar that has a spinner and tells us the elapsed time when it is
    finished.
    """
    def __init__(self, title='', **kw):
        self._pbar_widgets = [
                        title,
                        '  ',
                        progressbar.AnimatedMarker(),
                        '  ',
                        progressbar.FormatLabel(''),
                        '  ',
                        progressbar.ETA(
                            format_zero = '',
                            format_not_started='',
                            format_finished='%(elapsed)s elapsed',
                            format_NA='',
                            format='')]
        progressbar.ProgressBar.__init__(self, widgets=self._pbar_widgets, **kw)


    def finish(self,code):
        """
        Cleanup upon finishing
        """
        self._pbar_widgets[2] = ' ' # Delete the spinny wheel
        self._pbar_widgets[4] = self._finished_widget(code)
        progressbar.ProgressBar.finish(self)

    def _finished_widget(self, code):
        """
        Just a helper function to say our our progress went.
        """
        if code == 0:
            return progressbar.FormatLabel('[OK]')
        else:
            return progressbar.FormatLabel('[FAILED]')


class TitleElapsedProgressBar(progressbar.ProgressBar):
    """
    A progress bar that just tells us how long time has elasped upon
    completion.
    """
    def __init__(self, title='', **kw):
        self._pbar_widgets = [
            title,
            ' ',
            progressbar.ETA(
                format_zero = '',
                format_not_started = '',
                format_finished='%(elapsed)s elapsed',
                format_NA='',
                format=''
            )
        ]
        progressbar.ProgressBar.__init__(self, widgets=self._pbar_widgets, **kw)
