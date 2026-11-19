"""RGB PSNR and uniform-window SSIM for paired, equally sized images."""
import argparse,json
from pathlib import Path
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--references',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    import numpy as np
    from PIL import Image
    from skimage.metrics import structural_similarity
    def load(path):
        if path.suffix=='.npy':
            z=np.load(path,allow_pickle=False)
            if z.ndim==3 and z.shape[0]==3:z=z.transpose(1,2,0)
            if z.dtype==np.uint8:z=z.astype('float64')/255
        else:z=np.asarray(Image.open(path).convert('RGB'),dtype='float64')/255
        return np.asarray(z,dtype='float64').clip(0,1)
    preds=list(a.predictions.glob('*.npy')) or list(a.predictions.glob('*.png'))
    rows=[]
    for path in sorted(preds):
        reference=next((a.references/(path.stem+ext) for ext in ['.npy','.png','.tif','.tiff','.jpg'] if (a.references/(path.stem+ext)).exists()),None)
        if reference is None:raise FileNotFoundError(f'No reference with stem {path.stem}')
        x,y=load(path),load(reference)
        if x.shape!=y.shape or x.ndim!=3 or x.shape[2]!=3:raise ValueError(f'Invalid paired RGB dimensions for {path.stem}')
        mse=float(np.mean((x-y)**2))
        rows.append({'index':len(rows),'psnr_db':float(-10*np.log10(mse)) if mse else float('inf'),'ssim':float(structural_similarity(y,x,data_range=1,channel_axis=2,win_size=7,gaussian_weights=False,use_sample_covariance=True,K1=.01,K2=.03))})
    if not rows:raise ValueError('No predictions found')
    report={'n':len(rows),'metric':'full RGB, range [0,1], uniform 7x7 SSIM, sample covariance','mean':{k:float(np.mean([r[k] for r in rows])) for k in ['psnr_db','ssim']},'per_image':rows}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report['mean']))
if __name__=='__main__':main()
