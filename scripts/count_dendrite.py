import sys
from scipy.misc import imread
from em_count.emio import writeh5 

opt = sys.argv[1]
Dl='/n/boslfs/LABS/lichtman_lab/Donglai/jwr_yuelong/'
D0='/n/coxfs01/donglai/data/JWR/snow_cell/'
D1='/n/coxfs01/donglai/data/JWR/toufiq_seg/'

sz128_iso=[773,832,832] # 128x128x120nm

if opt[0]=='1': 
    if opt =='1': # output egg, 128nm
        fin = 'egg';fout='cell_yl_cb.h5'
        fin = 'eggdrop';fout='cell_yl_den.h5'
        Do = D0+'cell128nm/'+fout

        out=np.zeros(sz128_iso, dtype=np.uint8)
        for i in range(sz128_iso[0]):
            print i
            out[i,32:,32:] = imread(Dl+fin+'/%04d.png'%(4*i+1), 'L')[::2,::2]
        writeh5(Do, 'main', out)
    elif opt =='1.1': # cc
        # for visualization (hp03): python -i ng_pair_128nm.py cell_yl_den_cc_bv.h5
        from skimage.measure import label
        from scipy.ndimage.morphology import binary_opening,binary_dilation
        dopt = int(sys.argv[2])
        numI = int(sys.argv[3])
        numD = int(sys.argv[4])
        Do = D0+'cell128nm/'
        fn = 'cell_yl_den'
        seg = np.array(h5py.File(Do+fn+'.h5')['main'])

        suf='_bv'
        seg_db = np.array(h5py.File(Do+'cell_bv.h5')['main'])
        mm = seg_db>0 if numD==0 else binary_dilation(seg_db>0,iterations=numD)
        seg[mm] = 0 
        if dopt == 1:# cell mask from db
            seg_db = np.array(h5py.File(Do+'cell_daniel.h5')['main'])
            mm = seg_db>0 if numD==0 else binary_dilation(seg_db>0,iterations=numD)
            seg[mm] = 0 
            suf+='_db'
        writeh5(Do+fn+'_cc%s_open%d_dl%d.h5'%(suf,numI,numD), 'main', relabel(label(binary_opening(seg,iterations=numI)),do_sort=True))
