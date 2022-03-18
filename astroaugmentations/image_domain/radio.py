import numpy as np
import albumentations as A
from skimage.segmentation import slic

__all__ = [
    "SpectralIndex",
    "CustomKernelConvolution"
]

class SpectralIndex():
    """Augments images in accordance naturally occuring
    spectral index variations.
    Args:
        mean (float):
            Mean expected spectral index. Default -0.8.
        std (float):
            Spectral index standard deviation used to
            define normal curve to sample from. Default 0.2.
        super_pixels (bool):
            Whether or not to use super pixels and apply
            SpectralIndex augmentations independantly
            to each super pixel. Default False.
        n_segments (int):
            How many segments the image should be
            disected into.
        seed (int):
            Seed the random sampling (for debugging
            only, default None).
        compactness (float):
            Compactness of the segmentations. Supplied
            to skimage.SLIC call. Default 0.0001.
        segmentation_out (bool):
            Allows user to read out super_pixel segmentation
            map. Default False.
    """
    def __init__(self, mean=-0.8, std=0.2, super_pixels=False,
                 n_segments=225, seed=None, compactness=0.0001,
                 segmentation_out=False):
        self.mean = mean
        self.std = std
        self.super_pixels = super_pixels
        self.n_segments = n_segments
        self.compactness = compactness
        self.seed = seed
        self.segmentation_out = segmentation_out


    def __call__(self, image, **kwargs):
        augmented_image = image.copy()
        if self.seed is not None:
            np.random.RandomState.seed = self.seed
        if self.super_pixels:
            image = np.float32(image)
            segments = self.segment(image)
            segment_scaling = segments.copy()
            for seg_val in np.unique(segments):
                segment_scaling = np.where(
                    segments==seg_val, self.random_spectral_scaling(), segment_scaling)
            augmented_image = augmented_image * segment_scaling
        else:
            augmented_image = augmented_image * self.random_spectral_scaling()
        if self.segmentation_out:
            return augmented_image, segments
        else:
            return augmented_image

    def segment(self, image):
        segments = slic(
            image, n_segments=self.n_segments,
            slic_zero=True, compactness=self.compactness,
            start_label=1)
        return segments

    def random_spectral_scaling(self):
        sample = np.random.randn(1)[0]*self.std + self.mean
        pct_brightness_change = (self.mean - sample)/self.mean
        return 1. + pct_brightness_change

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
        psf_radius (float):
            Radius of psf centre to clip out. Default=None; FIRST=2.4 (resolution in arcsec).
        sidelobe_scaling (float):
            Factor to upweight sidelobes by, default: 1.
        mode (str):
            How to manage the kernel interacting with existing signal. Default: 'sum'.
            Options:
                'sum': Adds original image back onto convolved image.
                'masked': Masks non-zeros in convolved image
                    before adding back in kernel (can create sharp edges).
                'delta': Adds a delta function to centre of kernel
                    (can create signifcant artifacts from non-smooth kernel).
    """
    def __init__(self, kernel, rfi_dropout=None, rfi_dropout_p=1, beam_cut=None,
                 psf_radius=None, sidelobe_scaling=1, show_kernel=False, mode='sum'):
        self.mode = mode
        assert self.mode in ["masked", "sum", "delta"]
        self.kernel = kernel
        self.rfi_dropout = rfi_dropout
        self.rfi_dropout_p = rfi_dropout_p
        self.beam_cut = beam_cut
        self.psf_radius = psf_radius
        self.show_kernel=show_kernel
        self.sidelobe_scaling = sidelobe_scaling

    def __call__(self, image, **kwargs):
        # Match kernel size to image size
        kernel = np.zeros(image.shape, dtype=np.complex128)
        im_mid = [image.shape[0]//2, image.shape[1]//2]
        k_mid = [self.kernel.shape[0]//2, self.kernel.shape[1]//2]
        kernel[im_mid[0]-k_mid[0]:im_mid[0]+k_mid[0], im_mid[0]-k_mid[0]:im_mid[0]+k_mid[0]] = self.kernel

        # Adjust sidelobes to match expected noise level
        # Set central beam above self.beam_cut to 0
        if self.beam_cut is not None:
            kernel = self.beam_cutting(kernel)
        # Remove central pixels if within given pixel radius
        kernel = self.radius_clipping(kernel) if self.psf_radius is not None else kernel

        # Add delta function to sample original image
        if self.mode == "delta":
            kernel[kernel.shape[0]//2, kernel.shape[1]//2] = np.real(kernel).max()/self.max_sidelobe+1j*0

        # FTs
        ft_img = np.fft.fft2(image)
        ft_kernel = np.fft.fft2(kernel)

        # RFI Flagging:
        if self.rfi_dropout is not None:
            transform = A.Compose([A.PixelDropout(dropout_prob=self.rfi_dropout, p=self.rfi_dropout_p)])
            ft_kernel = transform(image=ft_kernel)['image']

        # Convolved image
        convolved_image = np.fft.fftshift(np.fft.ifft2(ft_img*ft_kernel))
        max = 1. if np.abs(convolved_image).max()==0 else np.abs(convolved_image).max()
        convolved_image = (np.abs(convolved_image)-np.abs(convolved_image).min())/max

        # Add back in original signal
        if self.mode=="masked":
            convolved_image = np.where(image>0, 0, np.abs(convolved_image))
            convolved_image = convolved_image*self.sidelobe_scaling + image
        elif self.mode=="sum":
            covolved_image = (np.abs(convolved_image)+image)/2
        return convolved_image

    def beam_cutting(self, kernel):
        kernel = np.where(np.abs(kernel)/np.abs(kernel).max()>self.beam_cut, 0, kernel)
        return kernel

    def radius_clipping(self, kernel):
        xx, yy = np.mgrid[0:kernel.shape[0], 0:kernel.shape[0]] - kernel.shape[0]//2
        rr = np.sqrt(xx**2+yy**2)
        kernel = np.where(rr<=self.psf_radius, 0, kernel)
        return kernel
