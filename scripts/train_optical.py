"""Portable bounded joint optical/decoder training on authorized image folders."""
import argparse,sys,json,random
from pathlib import Path
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--factor',type=int,choices=[12,16],required=True)
    p.add_argument('--train-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=2000)
    p.add_argument('--seed',type=int,default=20261106)
    a=p.parse_args()
    import numpy as np,torch,imgaug
    import torch.nn.functional as F
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src/optical'))
    from models.MIMOUNet_Aux import build_net
    from aligned_two_layer import AlignedTwoLens
    from data.data_load import train_dataloader
    a.output.mkdir(parents=True,exist_ok=False)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);imgaug.seed(a.seed)
    model=build_net('MIMO-UNet').cuda();lens=AlignedTwoLens().cuda()
    opt_m=torch.optim.Adam(model.parameters(),lr=1e-4)
    opt_l=torch.optim.Adam(lens.parameters(),lr=1e-2)
    loader=train_dataloader(str(a.train_dir),batch_size=2,num_workers=0,use_transform=True,psf_gene_obj=lens,shuffle=True)
    iterator=iter(loader)
    with (a.output/'training.jsonl').open('w') as log:
        for step in range(1,a.steps+1):
            try:batch=next(iterator)
            except StopIteration:iterator=iter(loader);batch=next(iterator)
            x,target,*_=batch;target=target.cuda()
            x=F.interpolate(x,scale_factor=1/a.factor,mode='bilinear',align_corners=False,recompute_scale_factor=False)
            x=F.interpolate(x,scale_factor=float(a.factor),mode='bilinear',align_corners=False,recompute_scale_factor=False)
            opt_m.zero_grad();opt_l.zero_grad();pred=model(x)
            labels=[F.interpolate(target,scale_factor=s,mode='bilinear',align_corners=False) for s in [.25,.5,1.]]
            l1=sum((v-t).abs().mean() for v,t in zip(pred,labels))
            freq=sum((torch.fft.fft2(v)-torch.fft.fft2(t)).abs().mean() for v,t in zip(pred,labels))
            loss=l1+.05*freq
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            loss.backward();opt_m.step();opt_l.step()
            log.write(json.dumps({'step':step,'loss':float(loss),'l1':float(l1),'fft':float(freq)})+'\n')
    torch.save({'model':model.state_dict(),'lens_module':lens.state_dict(),'step':a.steps,'factor':a.factor,'role':'portable joint-training run'},a.output/'model.pt')
    print(f'Completed {a.steps} joint optimizer updates')
if __name__=='__main__':main()
