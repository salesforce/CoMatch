import torchvision.transforms as transforms
    
class ThreeCropsTransform:
    """Take 3 random augmentations of one image."""

    def __init__(self,trans_weak,trans_strong0,trans_strong1):       
        self.trans_weak = trans_weak
        self.trans_strong0 = trans_strong0
        self.trans_strong1 = trans_strong1
    def __call__(self, x):
        x1 = self.trans_weak(x)
        x2 = self.trans_strong0(x)
        x3 = self.trans_strong1(x)
        return [x1, x2, x3]