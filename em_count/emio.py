import h5py
import numpy as np


def writetxt(filename, content):
    a= open(filename,'w')
    a.write(content)
    a.close()

def writeh5(filename, datasetname, dtarray):
    fid=h5py.File(filename,'w')
    if isinstance(datasetname, (list,)):
        for i,dd in enumerate(datasetname):
            ds = fid.create_dataset(dd, dtarray[i].shape, compression="gzip", dtype=dtarray[i].dtype)
            ds[:] = dtarray[i]
    else:
        ds = fid.create_dataset(datasetname, dtarray.shape, compression="gzip", dtype=dtarray.dtype)
        ds[:] = dtarray
    fid.close()

def U_mkdir(fn):
    if not os.path.exists(fn):
        os.mkdir(fn)


