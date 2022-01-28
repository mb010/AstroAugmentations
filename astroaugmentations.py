import albumentations as A
import numpy as np

class AstroAugmentations():
    """Return augmentation object designed for a specific regime & data type within astronomy."""
    def __init__(self, domain:str="default", data_type:str='imaging'):
        self.domain = domain
        self.data_type = data_type
        ### Composed example augmentations:
        self.augmentation = {
            "imaging": {
                "optical": A.Compose([]),
                "radio": A.Compose([]),
                "default": A.Compose([])
            }
        }
    
    def __call__(self, image):
        return self.augmentation[self.data_type][self.domain](image=image)['image']

class BrightnessGradient():
    """Randomly applies a linear gradient to the brightness of the image.
    Args:
        limits (tuple): 
            Upper and lower bound for uniform random selection of 
            brightness scaling limits of the applied gradient.
        primary_beam (bool): 
            Whether or not to emulate an incorrectly corrected primary beam.
        noise (float):
            Fraction of maximum signal at which the randomly selected
            gaussian noise should be capped.
        seed (float):
            Seed used for random numpy values.
    """
    def __init__(self, limits=(0.5,1), primary_beam:bool=False, seed=None, noise=0.2):
        self.seed = seed
        if self.seed is not None:
            np.random.RandomState.seed = self.seed
        self.noise = noise
        self.primary_beam = primary_beam
        self.limits = np.asarray(limits)
        
        
    def __call__(self, image, **kwargs):
        if self.primary_beam:
            return self.primary_beam_analogy(image)
        else:
            return self.gradient(image)
    
    def gradient(self, image):
        limits = self.limits[0] + np.diff(self.limits)[0]*np.sort(np.random.rand(2))
        grid = np.mgrid[0:image.shape[0], 0:image.shape[1]]
        normal = np.random.rand(2)*2-1
        scaling = (normal[0]*grid[0] - normal[1]*grid[1])
        scaling = scaling-scaling.min()+limits[0]
        transformed_image = image*scaling/scaling.max()*limits[1]
        print(limits)
        
        return transformed_image

    def primary_beam_analogy(self, image):
        transform = A.Compose([
            A.GaussNoise(var_limit=self.noise*image.max(), mean=0, always_apply=True,),
            A.Lambda(
                name="gradient", 
                image=BrightnessGradient(
                    limits=self.limits, 
                    primary_beam=False, 
                    seed=self.seed,
                    noise=self.noise),
                always_apply=True)
        ])
        return transform(image=image)['image']



class CustomKernelConvolution():
    """Creation of a function which convolves the input 
    image with a kernel (psf) as provided by the user.
    Args:
        img (nd.array): 
            2d Image to be convolved (or batch of images 
            as numpy array.
        kernel (nd.array, np.complex128): 
            Convolution kernel. Height and width must be smaller than or equal to image.
        rfi_dropout (float): 
            Percentage of pixels to drop in the kernels fourrier transformation.
        rfi_dropout_p (float): 
            Probability of applying rfi_dropout.
        beam_cut (float): 
            Kernel value above which to set kernel values to 1.
    Returns:
        convolved_image (nd.array):
            MinMax normalised convolved image.
    """
    def __init__(self, kernel, rfi_dropout=None, rfi_dropout_p=1, beam_cut=None, show_kernel=False):
        self.kernel = kernel
        self.rfi_dropout = rfi_dropout
        self.rfi_dropout_p = rfi_dropout_p
        self.beam_cut = beam_cut
    
    def __call__(self, image, **kwargs):
        # Match kernel size to image size
        kernel = np.zeros(image.shape, dtype=np.complex128)
        im_mid = [image.shape[0]//2, image.shape[1]//2]
        k_mid = [self.kernel.shape[0]//2, self.kernel.shape[1]//2]
        kernel[im_mid[0]-k_mid[0]:im_mid[0]+k_mid[0], im_mid[0]-k_mid[0]:im_mid[0]+k_mid[0]] = self.kernel
        
        # Adjust sidelobes to match expected noise level
        if self.beam_cut is not None:
            # Attempt 1: Set central beam to 1
            kernel = np.where(np.abs(kernel)/np.abs(kernel).max()>self.beam_cut, 1+1j*0, kernel)
            
            # Attempt 2: Cut and replace central pixel
            #kernel = np.where(np.abs(kernel)/np.abs(kernel).max()>self.beam_cut, 0, kernel)
            #kernel[kernel.shape[0]//2+1, kernel.shape[1]//2+1] = 1
            
            # Attempt 3: Remove central pixels if within first null region?

        # FTs
        ft_img = np.fft.fft2(image)
        ft_kernel = np.fft.fft2(kernel)

        # RFI Flagging:
        if self.rfi_dropout is not None:
            transform = A.Compose([A.PixelDropout(dropout_prob=self.rfi_dropout, p=self.rfi_dropout_p)])
            ft_kernel = transform(image=ft_kernel)['image']
        
        # Final image
        convolved_image = np.fft.fftshift(np.fft.ifft2(ft_img*ft_kernel))
        convolved_image = (np.abs(convolved_image)-np.abs(convolved_image).min())/np.abs(convolved_image).max()
        return convolved_image
