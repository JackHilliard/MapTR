import torch
from mmdet3d.models.builder import build_middle_encoder
cfg=dict(type='SparseEncoder', in_channels=4, sparse_shape=[301,301,421],
         output_channels=128, order=('conv','norm','act'),
         encoder_channels=((16,16,32),(32,32,64),(64,64,128),(128,128)),
         encoder_paddings=([0,0,1],[0,0,1],[0,0,[1,1,0]],[0,0]),
         block_type='basicblock')
m=build_middle_encoder(cfg).cuda().eval()
N=20000; B=2
z=torch.randint(0,420,(N,)); y=torch.randint(0,300,(N,)); x=torch.randint(0,300,(N,))
b=torch.randint(0,B,(N,))
coors=torch.stack([b,z,y,x],1).int().cuda(); feat=torch.randn(N,4).cuda()
with torch.no_grad(): out=m(feat,coors,B)
print('SparseEncoder out:', tuple(out.shape), '-> channels %d, spatial %dx%d'%(out.shape[1],out.shape[2],out.shape[3]))
