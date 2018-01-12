import matplotlib as mpl
from copy import copy


class BlitArtist:
    """
    A base class to give us methods we can use to initialize an object that
    we can "blit" and a method to update it over the course of animation.
    """

    def __init__(self, bdata, **kw):
        self._bdata = bdata
        self._artist_kw = kw
        self._ax_ndx = None
        self._build_kw = {}

    def blit_build(self, ax, ax_ndx=None, **kw):
        raise NotImplementedError('build must be implemented for a BlitPatch')

    def blit_update(self, frame, update, ax_ndx=None, **kw):
        raise NotImplementedError('blit_update must be implemented for a BlitPatch')



class BRectangle(mpl.patches.Rectangle, BlitArtist):
    """
    A BlitArtist version of the Rectangle Patch
    """

    def __init__(self, xy, width, height, bdata=None, imshow=True, **kw):
        """
        The initialize function matches that of patches.Rectangle with the
        exception of imshow, which is used to adjust offsets in heatmaps.

        :param xy: tuple of the left-lower corner (x,y)
        :param width: the width of the rectangle
        :param height: the height of the rectangle
        :param data: any additional data we'd like to keep around (e.g. for udpating)
        :param imshow: if true, adjust the rectangle to land on a heatmap boundary
        """
        BlitArtist.__init__(self, bdata, **kw)
        offset = 0 if not imshow else 0.5
        self._xy = xy[0]-offset, xy[1]-offset
        self._width = width
        self._height = height


    def blit_build(self, ax, ax_ndx=None, **kw):
        """
        Actually build the patch and add it to the axes.  This method gets called
        via the init_func in FuncAnimation.

        :param ax: the axes to add the patch to

        :return: ourself, since we're actually an artist
        """
        obj = self if ax_ndx is None else copy(self)
        mpl.patches.Rectangle.__init__(obj, obj._xy, obj._width, obj._height, **obj._artist_kw)
        ax.add_patch(obj)
        obj._ax_ndx = ax_ndx
        obj._build_kw == kw
        return obj

    def blit_update(self, frame, update, ax_ndx=None, **kw):
        """
        In the animation function, after the major elements are drawn,
        update this object in a specified manner.

        :param frame: the current frame number
        :param update: the update number associated with this frame
        :param ax_ndx: the index of the axes we're drawing on

        :return: none
        """
        pass


class BCircle(mpl.patches.Circle, BlitArtist):
    """
        A circular blit artist.
    """
    def __init__(self, xy, radius, bdata=None, imshow=True, **kw):
        BlitArtist.__init__(self, bdata, **kw)
        offset = 0 if not imshow else 0.5
        self._xy = xy[0]-offset, xy[1]-offset
        self._radius = radius

    def blit_build(self, ax, ax_ndx=None, **kw):
        obj = self if ax_ndx is None else copy(self)
        mpl.patches.Circle.__init__(obj, obj._xy, obj._radius, **obj._artist_kw)
        ax.add_patch(obj)
        obj._ax_ndx = ax_ndx
        obj._build_data == kw
        return obj

    def blit_update(self, frame, update, ax_ndx=None, **kw):
        pass


class BAnnotation(mpl.text.Annotation, BlitArtist):
    """
    An annotation blit artist
    """
    def __init__(self, bdata=None, **kw):
        BlitArtist.__init__(self, bdata, **kw)

    def blit_build(self, ax, ax_ndx=None, **kw):
        obj = self if ax_ndx is None else copy(self)
        mpl.text.Annotation.__init__(obj, **obj._artist_kw)
        ax.add_artist(obj)
        return obj



class BCellHighlighter(mpl.collections.PatchCollection, BlitArtist):
    """
    A class that overlays a cell grid on top of the axes.  The update
    method can be implmented to highlight particular cells.
    """
    def __init__(self, gridshape, bdata=None, imshow=True, **kw):
        BlitArtist.__init__(self, bdata, **kw)
        self._imshow = imshow

        self._gridshape = gridshape

    def blit_build(self, ax, ax_ndx=None, **kw):
        obj = self if ax_ndx is None else copy(self)
        x,y = obj._gridshape
        patches = []
        for r in range(y):
            for c in range(x):
                offset = 0 if not obj._imshow else 0.5
                xx = r-offset
                yy = c-offset
                patch = mpl.patches.Rectangle((xx,yy), width=1, height=1, **obj._artist_kw)
                patches.append(patch)
        mpl.collections.PatchCollection.__init__(obj, patches)
        ax.add_collection(obj)
        obj.set_edgecolors('none')
        obj.set_facecolors('none')
        return obj
