import torch
from models.MIMOUNet_Aux import LensModule
class AlignedTwoLens(LensModule):
 def get_psfs(self,*args,**kwargs):
  result=super().get_psfs(*args,**kwargs)
  with torch.no_grad():
   torch.rand_like(self.lens_layer1)
  return result
