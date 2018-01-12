import numpy as np
cimport numpy as np
cimport cython
import pdb

DTYPE = np.float
ctypedef np.float_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray blender(np.ndarray[DTYPE_t, ndim=2] data, np.ndarray[DTYPE_t, ndim=3] lscmap):

  #Some indexers
  cdef int i,j,k,r,c,_,__, cell, color_ndx, cmap_level
  cdef float this_color, sum_color
  cdef int ud_start, ud_end
  cdef int res_id, res_ndx
  cdef int ud_curr
  cdef int row_curr

  cdef int num_cells = data.shape[1] - 2
  cdef int num_rows = data.shape[0]
  cdef int num_res
  cdef int ndx_u, ndx_r
  cdef int col_ud = 0
  cdef int col_res = 1
  cdef int col_cells = 2

  cdef int num_updates = len(np.unique(data[:,col_ud]))

  # We're assuming all colormaps have the same number of levels
  cdef int num_levels
  num_res = lscmap.shape[0]
  num_levels = lscmap.shape[1]
  cdef float scale = 1.0/num_res

  # Get our mins and maxes so we can normalize all our data
  cdef double vmin = np.min(data[0,col_cells:])
  cdef double vmax = np.max(data[0,col_cells:])
  for k in range(1, num_rows):
    _max = np.max(data[k,col_cells:])
    _min = np.min(data[k,col_cells:])
    if _min < vmin:
        vmin = _min
    if _max > vmax:
        vmax = _max

  # Normalize our data
  # Matplotlib's normalizing routine makes sure that the maximum value
  # of all normalized data is in the range of [0,1) by altering the
  # maximum value to a floating point just short of 1.0
  cdef float near_one=np.nextafter(1,0,dtype=DTYPE)
  cdef float val
  for r in range(num_rows):
    for c in range(num_cells):
      val = (data[r,col_cells+c] - vmin) / (vmax-vmin)
      data[r,col_cells+c] = val if val != 1.0 else near_one


  # This will be our blended output with update in the first dimension,
  # cell array in the second dimension, and rgb in the third.
  # I'm not doing alpha blending.
  cdef np.ndarray[DTYPE_t, ndim=3] u_rgb = np.ones([num_updates, num_cells, 3], dtype=DTYPE)

  # Iterate through our data.  I'm assuming the data is sorted by update
  # already.
  ndx_u = 0
  row_curr = 0
  while row_curr < num_rows:
    # Find the beginning and ending indicies for the update
    # Each resource has its own row with the same update number
    ud_start = row_curr
    ud_end = ud_start
    ud_curr = <int>data[row_curr,col_ud]
    while ud_curr == <int>data[row_curr,col_ud] and row_curr < num_rows:
      row_curr += 1
    ud_end = row_curr

    # Do our blending
    for cell in range(num_cells):
        for r in range(ud_start, ud_end):
          res_id = <int>data[r,col_res]
          cmap_level = <int>(data[r,cell+col_cells] * num_levels)
          for color_ndx in range(3):
            this_color = lscmap[res_id,cmap_level,color_ndx]
            sum_color = u_rgb[ndx_u,cell,color_ndx]
            u_rgb[ndx_u,cell,color_ndx] = min(this_color, sum_color)

    ndx_u += 1

  return u_rgb
