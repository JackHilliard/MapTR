import torch, numpy as np, glob
from mmdet3d.models.builder import build_middle_encoder
def probe(sparse_shape, hxy, label):
    cfg=dict(type='SparseEncoder', in_channels=4, sparse_shape=sparse_shape,
             output_channels=128, order=('conv','norm','act'),
             encoder_channels=((16,16,32),(32,32,64),(64,64,128),(128,128)),
             encoder_paddings=([0,0,1],[0,0,1],[0,0,[1,1,0]],[0,0]),
             block_type='basicblock')
    m=build_middle_encoder(cfg).cuda().eval()
    N=40000; B=2
    a=torch.randint(0,hxy,(N,)); b_=torch.randint(0,hxy,(N,)); z=torch.randint(0,420,(N,))
    bi=torch.randint(0,B,(N,))
    coors=torch.stack([bi,a,b_,z],1).int().cuda(); feat=torch.randn(N,4).cuda()
    with torch.no_grad(): out=m(feat,coors,B)
    cell=30.0/out.shape[2]
    print('%-22s sparse_shape=%s -> out %s   BEV %dx%d = %.3f m/cell'%(
        label, sparse_shape, tuple(out.shape), out.shape[2], out.shape[3], cell))
    del m; torch.cuda.empty_cache()
probe([301,301,421],300,'baseline v=0.10')
probe([601,601,421],600,'v=0.05 (2x)')
# how many voxels does a real crop actually produce at each voxel size?
fs=sorted(glob.glob('/data_carla50/test/blocks/*.npz'))[:8]
print()
for vs in (0.10,0.05):
    tot=[]
    for f in fs:
        p=np.load(f,allow_pickle=True)['features'][:,:3]
        m_=(np.abs(p[:,0])<15)&(np.abs(p[:,1])<15); q=p[m_]
        k=np.unique(np.floor(q/np.array([vs,vs,0.4])).astype(np.int64),axis=0)
        tot.append(len(k))
    print('  voxel %.2f m -> %d voxels/crop (median)   [config max_voxels 90000/120000]'%(vs,int(np.median(tot))))
