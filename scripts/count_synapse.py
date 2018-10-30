import os,sys
import numpy as np
import h5py
from skimage.measure import label

from em_count.emio import *
from em_count.bbox import *

opt = sys.argv[1]
# opt=0: create folder
# opt=1: find bboxes for each prediction in different volumes
# opt=2: bboxes = _in + _bd
# opt=3: connect xy: _bd -> _bd_xy
# opt=4: _bd_xy = _bd_xy_zin + _bd_xy_zbd
# opt=5: connect z _bd_xy_zbd = _bd_xy_zbd_z
# opt=6: thres bbox by size

Do='/n/coxfs01/donglai/ppl/zudi/jwr/syn_test_0904/'
D0 = '/n/coxfs01/zudilin/research/synapseNet/data/jwr100/synapse_round_1/'
sz = (96,1120,1120);
xx = range(0,11200+1,sz[2]) # 11 
yy = xx
zz = range(0,2976+1,sz[0]) # 32
pad=[8,8]
chunk=[1,1,1]
jn = '20um_20180720/bfly_v2-2_add8.json';suf='_jwr'
# patch: 1120x1120x96
# computation: 3093, add_8_len: 3109
tile_sz_im=2560;tile_r=1
slice_sz_im=tile_sz_im*5
numZ=3109

if opt == '0': # create folder
    DD = Do[:-1]+'_txt/' # for txt_vol
    DD = Do # for txt
    for z in zz:
        os.mkdir(DD+str(z))
elif opt == '1': # find bbox
    jobId = int(sys.argv[2])
    jobNum = int(sys.argv[3])
    cc = 0
    thres=128
    redo = 1
    for x in xx:
        for y in yy:
            for z in zz:
                if cc % jobNum == jobId:
                    print x,y,z
                    fn = D0+'%d/%d/%d/mask.h5'%(x,y,z)
                    sn = Do+'%d/%d_%d_%d_%d_%d.txt'%(z,chunk[0],x,y,chunk[2],chunk[1])
                    if redo==0 and os.path.exists(sn):
                        continue
                    data=h5py.File(fn)['main'][2]
                    delta = np.zeros((7))
                    if z==zz[0]:
                        data[:pad[0]] = 0
                    if z==zz[-1]:
                        if z+sz[0]>numZ-pad[1]:
                            data[(numZ-pad[1])-(z+sz[0]):] = 0
                    # if h5 is big: break it up to chunks and combine them
                    for zi in range(chunk[0]):
                        for yi in range(chunk[1]):
                            for xi in range(chunk[2]):
                                sn2 = Do+'%d/%d_%d_%d_%d_%d.txt'%(z,zi,x,y,xi,yi)
                                if redo==0 and os.path.exists(sn2):
                                    continue
                                print xi,yi,zi
                                mm = label(np.array(data[rr[0][zi]:rr[0][zi+1],
                                                         rr[1][yi]:rr[1][yi+1],
                                                         rr[2][xi]:rr[2][xi+1]])>thres)
                                uid = np.unique(mm)
                                uid = uid[uid>0]
                                # check bbox
                                bb=np.zeros((len(uid),7), np.uint16)
                                for i in range(len(uid)):
                                    bb[i] = get_bb(mm==uid[i], True)
                                np.savetxt(sn2, bb,'%d')
                cc += 1
elif opt == '2': # divide boxes into inner and border
    jobId = int(sys.argv[2])
    jobNum = int(sys.argv[3])
    cc = 0
    zz2=zz
    #zz2=[0]
    for x in xx:
        for y in yy:
            for z in zz2:
                if cc % jobNum == jobId:
                    s1 = Do+'%d/%d_%d_in.txt'%(z,x,y)
                    s2 = Do+'%d/%d_%d_bd.txt'%(z,x,y)
                    # read in all sub-vol
                    # bbox: ready for union
                    bb = bbox_loadM(chunk,rr,Do+'%d/'%(z)+'%d_'+'%d_%d'%(x,y)+'_%d_%d.txt')
                    # xy connect
                    for zi in range(chunk[0]):
                        # link lr (y)
                        #print bb[zi][0][0].shape[0],bb[zi][0][1].shape[0],bb[zi][1][0].shape[0],bb[zi][1][1].shape[0]
                        for xi in range(chunk[2]):
                            for yi in range(chunk[1]):
                                if yi==chunk[1]-1:
                                    continue
                                #print 'in',zi,xi,yi,bb[zi][yi][xi].shape,bb[zi][yi+1][xi].shape
                                bb[zi][yi][xi],bb[zi][yi+1][xi] = bbox_link(bb[zi][yi][xi],bb[zi][yi+1][xi],
                                                                    3,2,np.array([0,1,4,5]),
                                                                    rr[1][yi+1]-1,rr[1][yi+1])
                                #print 'out',zi,xi,yi,bb[zi][yi][xi].shape,bb[zi][yi+1][xi].shape

                        #print bb[zi][0][0].shape[0],bb[zi][0][1].shape[0],bb[zi][1][0].shape[0],bb[zi][1][1].shape[0]
                        # link ud (x)
                        for xi in range(chunk[2]):
                            for yi in range(chunk[1]):
                                if xi==chunk[2]-1:
                                    continue
                                #print 'in',zi,xi,yi,bb[zi][yi][xi].shape,bb[zi][yi][xi+1].shape
                                bb[zi][yi][xi],bb[zi][yi][xi+1] = bbox_link(bb[zi][yi][xi],bb[zi][yi][xi+1],
                                                                    5,4,np.array([0,1,2,3]),
                                                                    rr[2][xi+1]-1,rr[2][xi+1])
                                #print 'out',zi,xi,yi,bb[zi][yi][xi].shape,bb[zi][yi][xi+1].shape

                        print bb[zi][0][0].shape[0],bb[zi][0][1].shape[0],bb[zi][1][0].shape[0],bb[zi][1][1].shape[0]
                        bb[zi] = bbox_concate(bb[zi])
                    # z connect
                    for zi in range(chunk[0]-1):
                        #print 'in',zi,bb[zi].shape,bb[zi+1].shape
                        bb[zi],bb[zi+1] = bbox_link(bb[zi],bb[zi+1],1,0,np.array([2,3,4,5]),
                                                            rr[0][zi+1]-1,rr[0][zi+1])
                        #print 'out',zi,bb[zi].shape,bb[zi+1].shape
                   
                    bb = bbox_concate(bb)
                    #import pdb; pdb.set_trace()
                    bid = []
                    # if on tissue border, count it as in
                    # z-border
                    if z>zz[0]:
                        bid+=list(np.where(bb[:,0]==0)[0])
                    if z<zz[-1]:
                        bid+=list(np.where(bb[:,1]==sz[0]-1)[0])
                    # col-border
                    if y>yy[0]:
                        bid+=list(np.where(bb[:,2]==0)[0])
                    if y<yy[-1]:
                        bid+=list(np.where(bb[:,3]==sz[1]-1)[0])
                    if x>xx[0]:
                        bid+=list(np.where(bb[:,4]==0)[0])
                    if x<xx[-1]:
                        bid+=list(np.where(bb[:,5]==sz[2]-1)[0])

                    bid = np.unique(bid)
                    np.savetxt(s1, bb[np.in1d(range(bb.shape[0]),bid,invert=True)],'%d')
                    np.savetxt(s2, bb[bid],'%d')
                cc += 1
elif opt == '3': # connect bd bbox: make xy bbox
    jobId = int(sys.argv[2])
    jobNum = int(sys.argv[3])
    zz2=zz
    #zz2=[0]
    cc = 0
    for z in zz2:
        if cc % jobNum == jobId:
            sn = Do+'%d/%d_%d_bd_xy.txt'%(z,xx[-1],yy[-1])
            if True:#not os.path.exists(sn):
                # load all bbox: original coord 
                print z
                # load global coordinate
                bb = bbox_loadM(chunk_xy,rr_xy,Do+'%d/'%(z)+'%d_%d_bd.txt',bbN_xy) 
                # xy connect
                # link lr (y)
                for xi in range(chunk_xy[1]):
                    for yi in range(chunk_xy[0]):
                        if yi==chunk_xy[0]-1:
                            continue
                        print 'in',xi,yi,bb[yi][xi].shape,bb[yi+1][xi].shape
                        bb[yi][xi],bb[yi+1][xi] = bbox_link(bb[yi][xi],bb[yi+1][xi],
                                                            3,2,np.array([0,1,4,5]),
                                                            rr_xy[0][yi+1]-1,rr_xy[0][yi+1])
                        print 'out',xi,yi,bb[yi][xi].shape,bb[yi+1][xi].shape
                # link ud (x)
                for xi in range(chunk_xy[1]):
                    for yi in range(chunk_xy[0]):
                        if xi==chunk_xy[1]-1:
                            continue
                        print 'in',xi,yi,bb[yi][xi].shape,bb[yi][xi+1].shape
                        bb[yi][xi],bb[yi][xi+1] = bbox_link(bb[yi][xi],bb[yi][xi+1],
                                                            5,4,np.array([0,1,2,3]),
                                                            rr_xy[1][xi+1]-1,rr_xy[1][xi+1])
                        print 'out',xi,yi,bb[yi][xi].shape,bb[yi][xi+1].shape

                # save result
                for yi in range(len(yy)):
                    for xi in range(len(xx)):
                        xo = xi*sz[2]
                        yo = yi*sz[1]
                        delta = np.array([0,0,yo,yo,xo,xo,0])
                        sn = Do+'%d/%d_%d_bd_xy.txt'%(z,xx[xi],yy[yi])
                        # save local coordinate
                        np.savetxt(sn, bb[yi][xi]-delta,'%d')
        cc += 1
elif opt == '4': # bd_xy -> bd_xy_zin+bd_xy_zbd
    jobId = int(sys.argv[2])
    jobNum = int(sys.argv[3])
    zz2=zz
    #zz2=[0]
    cc = 0
    for x in xx:
        for y in yy:
            for z in zz2:
                print x,y,z
                if cc % jobNum == jobId:
                    fn = Do+'%d/%d_%d_bd_xy.txt'%(z,x,y)
                    s1 = Do+'%d/%d_%d_bd_xy_zin.txt'%(z,x,y)
                    s2 = Do+'%d/%d_%d_bd_xy_zbd.txt'%(z,x,y)
                    if True: #not os.path.exists(s2):
                        bb = bbox_load(fn,' ',int)
                        bid = []
                        # z-border
                        if z>zz[0]:
                            bid+=list(np.where(bb[:,0]==0)[0])
                        if z<zz[-1]:
                            bid+=list(np.where(bb[:,1]==sz[0]-1)[0])
                        bid = np.unique(bid)
                        np.savetxt(s1, bb[np.in1d(range(bb.shape[0]),bid,invert=True)],'%d')
                        np.savetxt(s2, bb[bid],'%d')
                cc += 1
elif opt == '5': # connect z:
    bb_l=None
    zz2=zz
    #zz2=[0]
    # need to be sequential
    zz_delta=np.array([sz[0],sz[0],0,0,0,0,0])
    for zid in range(len(zz)-1):
        if not zz[zid] in zz2:
            continue
        print zid
        if zid==0: # first volume
            bb_l = bbox_loadM(chunk_xy,rr_xy,Do+'%d/'%(zz[zid])+'%d_%d_bd_xy_zbd.txt',bbN_xy)
            bb_l = bbox_concate(bb_l)
        else:
            bb_l=bb_r.copy()

        bb_r = bbox_loadM(chunk_xy,rr_xy,Do+'%d/'%(zz[zid+1])+'%d_%d_bd_xy_zbd.txt',bbN_xy)
        bb_r = bbox_concate(bb_r)
        print "input: ",bb_r.shape
        # bbox: ready for union
        bb_l,bb_r = bbox_link(bb_l,bb_r+zz_delta,1,0,np.array([2,3,4,5]),
                                    sz[0]-1,sz[0]) 
        bb_r = bb_r-zz_delta
        print "output: ",bb_r.shape
        # output bb_l: back in same folder-org
        for yid in range(len(yy)):
            for xid in range(len(xx)):
                sn = Do+'%d/%d_%d_bd_xy_zbd_z.txt'%(zz[zid],xx[xid],yy[yid])
                yo = yid*sz[1]
                xo = xid*sz[2]
                yo2 = (yid+1)*sz[1]
                xo2 = (xid+1)*sz[2]
                pp = np.logical_and(bb_l[:,2]>=yo,bb_l[:,2]<yo2)
                pp = np.logical_and(pp,np.logical_and(bb_l[:,4]>=xo,bb_l[:,4]<xo2))
                bid=np.where(pp)
                delta = np.array([0,0,yo,yo,xo,xo,0])
                np.savetxt(sn, bb_l[bid]-delta,'%d')
    # output bb_r
    for yid in range(len(yy)):
        for xid in range(len(xx)):
            sn = Do+'%d/%d_%d_bd_xy_zbd_z.txt'%(zz[-1],xx[xid],yy[yid])
            yo = yid*sz[1]
            xo = xid*sz[2]
            yo2 = (yid+1)*sz[1]
            xo2 = (xid+1)*sz[2]
            pp = np.logical_and(bb_r[:,2]>=yo,bb_r[:,2]<yo2)
            pp = np.logical_and(pp,np.logical_and(bb_r[:,4]>=xo,bb_r[:,4]<xo2))
            bid=np.where(pp)
            delta = np.array([0,0,yo,yo,xo,xo,0])
            np.savetxt(sn, bb_r[bid]-delta,'%d')
elif opt == '6': # thres inner bbox
    jobId = int(sys.argv[2])
    zz2=zz
    #zz2=[0]
    thres = 300
    if jobId==0:
        fin='in';fout='in_%d'%(thres)
    elif jobId==1:
        fin='bd_xy_zin';fout='bd_xy_zin_%d'%(thres)
    elif jobId==2:
        fin='bd_xy_zbd_z';fout='bd_xy_zbd_z_%d'%(thres)
    for x in xx:
        for y in yy:
            for z in zz2:
                print x,y,z
                s1 = Do+'%d/%d_%d_%s.txt'%(z,x,y,fin)
                s2 = Do+'%d/%d_%d_%s.txt'%(z,x,y,fout)
                if not os.path.exists(s1):
                    print 'not exist:',s1
                    continue
                bb = np.loadtxt(s1,delimiter=' ')
                if len(bb)==0:
                    np.savetxt(s2, np.zeros(0))
                else:
                    bb=bb.astype(int)
                    if bb.ndim==1:
                        bb=bb.reshape((1,len(bb)))
                    np.savetxt(s2, bb[bb[:,-1]>=thres],'%d')

