import numpy as np

# bbox utility
def get_bbs(data, thres, chunk=[1,1,1]):
    rr=[range(0,data.shape[x],np.ceil(data.shape[x]/float(chunk[x]))) for x in range(3)]
    for x in range(3):
        if rr[x][-1] != data.shape[x]:
            rr[x] += [data.shape[x]]

    # bbox 
    bb = [[[None for z in range(chunk[0])] for y in range(chunk[1])] for x in range(chunk[2])]
    # slice 
    sxy = [[[None for z in range(chunk[0])] for y in range(chunk[1]-1)] for x in range(chunk[2]-1)]
    # initial bbox
    for zi,zz in enumerate(rr[0][:-1]):
        for yi,yy in enumerate(rr[1][:-1]):
            for xi,xx in enumerate(rr[2][:-1]):
                mm = label(np.array(data[rr[0][zi]:rr[0][zi+1],rr[1][yi]:rr[1][yi+1],rr[2][xi]:rr[2][xi+1]])>thres)
                uid = np.unique(mm)
                uid = uid[uid>0]
                # check bbox
                bb[zi][yi][xi] = np.zeros((len(uid),7), np.uint16)
                bb[zi][yi][xi] = np.zeros((len(uid),7), np.uint16)
                for i in range(len(uid)):
                    bb[zi][yi][xi][i] = get_bb(mm==uid[i], True)

def list_create(chunk):
    if len(chunk)==1:
        out = [None for zi in range(chunk[0])]
    elif len(chunk)==2:
        out = [[None for yi in range(chunk[1])] for zi in range(chunk[0])]
    elif len(chunk)==3:
        out = [[[None for xi in range(chunk[2])] for yi in range(chunk[1])] for zi in range(chunk[0])]
    return out

def bbox_load(fn,delim=' ',dtype=int):
    bb = np.loadtxt(fn,delimiter=delim)
    if bb.ndim==1:
        bb=bb.reshape((1,len(bb)))
    if len(bb)>0:
        bb=bb.astype(dtype)
    return bb


def bbox_loadM(chunk,rr,fn, bbN=None, delim=' ',dtype=int):
    bb = list_create(chunk)
    if len(chunk)==3:
        # load 3D
        for xi in range(chunk[2]):
            for yi in range(chunk[1]):
                for zi in range(chunk[0]):
                    if bbN is None:
                        tmp = bbox_load(fn%(zi,xi,yi),delim,dtype) 
                    else:
                        tmp = bbox_load(fn%(bbN[0][zi],bbN[2][xi],bbN[1][yi]),delim,dtype) 
                    if len(tmp)==0:
                        continue
                    if rr is not None:
                        zo = rr[0][zi]
                        yo = rr[1][yi]
                        xo = rr[2][xi]
                        tmp += np.array([zo,zo,yo,yo,xo,xo]+[0]*(tmp.shape[1]-6))
                    bb[zi][yi][xi] = tmp
    elif len(chunk)==2:
        # load 2D
        for xi in range(chunk[1]):
            for yi in range(chunk[0]):
                if bbN is None:
                    tmp = bbox_load(fn%(xi,yi),delim,dtype) 
                else:
                    tmp = bbox_load(fn%(bbN[1][xi],bbN[0][yi]),delim,dtype) 
                if len(tmp)==0:
                    continue
                if rr is not None:
                    yo = rr[0][yi]
                    xo = rr[1][xi]
                    tmp += np.array([0,0,yo,yo,xo,xo]+[0]*(tmp.shape[1]-6))
                bb[yi][xi] = tmp
    return bb

def bbox_concate(bb):
    if not isinstance(bb[0], (list,)):
        # 1D list
        out=np.zeros((0,bb[0].shape[1]),dtype=bb[0].dtype)
        for xx in bb:
            out=np.vstack([out,xx])
    else:
        if not isinstance(bb[0][0], (list,)):
            # 2D list
            out=np.zeros((0,bb[0][0].shape[1]),dtype=bb[0][0].dtype)
            for xx in bb:
                for yy in xx:
                    out=np.vstack([out,yy])
    return out

def bbox_link(bb_l,bb_r,ax_l,ax_r,ax_m,tt_l,tt_r):
    # bbox in the same coord
    # link bb_l/bb_r by ax_l/ax_r dim with threshold value t1/t2
    if min(len(bb_l),len(bb_r))==0:
        return bb_l,bb_r
    b1 = np.where(bb_l[:,ax_l]==tt_l)[0]
    b2 = np.where(bb_r[:,ax_r]==tt_r)[0] 
    if min(len(b1),len(b2))==0:
        return bb_l,bb_r

    # coord
    ax_u = np.array(sorted([ax_l,ax_r]+list(ax_m)))
    # val
    ax_v = np.array(list(set(range(bb_l.shape[1]))-set(ax_u)),dtype=ax_u.dtype)

    for j in b1:
        sc = get_area(bb_l[j,ax_m],bb_r[b2][:,ax_m])
        if sc.max()>0: # there's a merge
            sid = b2[np.argmax(sc)]
            #print "in:",bb_l[j], bb_r[sid]
            bb_l[j,ax_u] = get_union(bb_l[j,ax_u], bb_r[sid,ax_u])
            bb_l[j,ax_v] = bb_l[j,ax_v]+bb_r[sid,ax_v]
            #print "out:",bb_l[j]
            #import pdb; pdb.set_trace()
            bb_r[sid,:] = -1
    return bb_l, bb_r[np.where(bb_r[:,0]>=0)[0]]

def get_bb(seg, do_count=False):
    dim = len(seg.shape)
    a=np.where(seg>0)
    if len(a)==0:
        return [-1]*dim*2
    out=[]
    for i in range(dim):
        out+=[a[i].min(), a[i].max()]
    if do_count:
        out+=[len(a[0])]
    return out

def get_area(a,b):
    # n*6
    # a: one box
    # b: multiple box
    #[xmin,xmax,ymin,ymax]
    if b.ndim==1:
        b=b.reshape(1,b.shape[0])
    dd = np.ones(b.shape[0])
    for i in range(len(a)//2):
        dd = dd*np.maximum(0,np.minimum(a[i*2+1], b[:,i*2+1]) - np.maximum(a[i*2], b[:,i*2]))
    return dd

def get_union(a,b):
    #[xmin,xmax,ymin,ymax]
    ll=len(a)
    out=[None]*ll
    for i in range(0,ll,2):
        out[i] = min(a[i],b[i])
    for i in range(1,ll,2):
        out[i] = max(a[i],b[i])
    return out

def get_intersect(a,b):
    #[xmin,xmax,ymin,ymax]
    ll=len(a)
    out=[None]*ll
    for i in range(0,ll,2):
        out[i] = max(a[i],b[i])
    for i in range(1,ll,2):
        out[i] = min(a[i],b[i])
    return out
