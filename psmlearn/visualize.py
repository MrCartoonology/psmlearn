import numpy as np

def make_grid(imgs, ncols=None, verbose=False, idx2bbox=None):
    old_len = len(imgs)
    imgs = [img for img in imgs if img is not None]
    if len(imgs) < old_len:
        print("WARNING: make_grid: %d of %d imgs are None" % 
              (old_len-len(imgs), old_len))
    if len(imgs)==0:
        print("ERROR: returning empty grid")
        return np.zeros((10,10))

    img_rows = [img.shape[0] for img in imgs]
    img_cols = [img.shape[1] for img in imgs]
    img_mxrow = max(img_rows)
    img_mxcol = max(img_cols)

    if verbose:
        print('make_grid: max dims for these imgs: %d %d' % (img_mxrow, img_mxcol))
    grid_ncols = int(np.ceil(np.sqrt(len(img_rows))))
    if ncols is not None:
        grid_ncols = ncols
    grid_nrows = int(np.ceil(len(img_rows)/float(grid_ncols)))

    grid = np.zeros((grid_nrows*img_mxrow + 3*(grid_nrows-1), 
                     grid_ncols*img_mxcol + 3*(grid_ncols-1)), dtype=np.float32)
    
    for num, img, ir, ic in zip(range(len(imgs)), imgs, img_rows, img_cols):
        row = num // grid_ncols
        col = num % grid_ncols
        row0 = row * img_mxrow + row*3
        col0 = col * img_mxcol + col*3
        grid[row0:(row0+ir),col0:(col0+ic)] = img[:]
        if idx2bbox is not None:
            idx2bbox[num]=np.array([row0,row0+img_mxrow,col0,col0+img_mxcol])
    return grid
