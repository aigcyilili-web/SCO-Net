"""Convert optical optimization variables to height and wrapped phase maps."""
import argparse
from pathlib import Path
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    import numpy as np
    if a.weights.suffix=='.npz':
        with np.load(a.weights,allow_pickle=False) as f:state={k:f[k] for k in f.files}
    else:
        import torch
        state={k:v.detach().cpu().numpy() for k,v in torch.load(a.weights,map_location='cpu',weights_only=True)['lens_module'].items()}
    output={'wavelength_m':np.array([640,550,460])*1e-9,'refractive_index':np.array([1.6983,1.7091,1.7302]),'sample_pitch_m':np.array(3e-6)}
    for i in [1,2]:
        h=np.tanh(state[f'lens_layer{i}'])*1e-6;h=h-h.min()
        phase=2*np.pi*h[...,None]*(output['refractive_index']-1)/output['wavelength_m']
        output[f'height_{i}_m']=h
        output[f'phase_{i}_rad']=np.angle(np.exp(1j*phase)).astype('float32')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output,**output)
    print('Exported two height maps and three-channel wrapped phase maps')
if __name__=='__main__':main()
