import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage.segmentation import slic

class AstroAugmentations():
    """Return augmentation object designed for a
    specific regime & data type within astronomy.
    Args:
        domain (str):
            Which domain the augmentation is for.
            Currntly available options: "radio".
            Not implemented yet: "optical", "default" (CV).
        kernel (np.ndarray):
            A kernel of your choice used for the
            CustomConvolution in the radio augmentation
            (2d numpy array, Complex128).
        p (flaoat):
            Probability of each of the augmentations to
            be applied, default: 0.5.
        tensor_out (bool):
            Ask the final product to be output as a
            pytorch tensor.
        aug_no (int):
            Used to debug the compound transformations.
            Will only apply the first aug_no
            transformations if given.
    """
    def __init__(self, domain:str="default", kernel:np.ndarray=None, p:float=0.5, tensor_out:bool=True, aug_no:int=None):
        self.domain = domain
        self.tensor_out = tensor_out
        self.p = p
        self.aug_no = aug_no

        self.kernel = kernel

    def __call__(self):
        if self.domain=='radio':
            transformation = self.radio()
        elif self.domain == 'optical':
            transformation = self.optical()
        else:
            transformation = self.default()
        return transformation

    def optical(self):
        return A.Compose([])

    def radio(self):
        augmentations = [
            # Change source perspective
            A.Lambda(
                name="Superpixel spectral index change",
                image=SpectralIndex(
                    mean=-0.8, std=0.2, super_pixels=True,
                    n_segments=100, seed=None),
                p=self.p), # With segmentation
            A.Lambda(
                name="Brightness perspective distortion",
                image=BrightnessGradient(limits=(0.,1.)),
                p=self.p), # No noise
            A.ElasticTransform( # Elastically transform the source
                sigma=100, alpha_affine=25, interpolation=1,
                border_mode=1, value=0,
                p=self.p),
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.1,
                rotate_limit=90, interpolation=2,
                border_mode=0, value=0,
                p=self.p),
            A.VerticalFlip(
                p=0.5),
            # Change properties of noise / imaging artefacts
            A.Lambda(
                name="Spectral index change of whole image",
                image=SpectralIndex(
                    mean=-0.8, std=0.2, seed=None),
                p=self.p), # Across the whole image
            A.Emboss(
                alpha=(0.2,0.5), strength=(0.2,0.5),
                p=self.p), # Quick emulation of incorrect w-kernels # Doesnt force the maxima to 1
            A.Lambda(
                name='Dirty beam convlolution',
                image=CustomKernelConvolution(
                    kernel=self.kernel, rfi_dropout=0.4, psf_radius=1.3,
                    sidelobe_scaling=1, mode='sum'),
                p=self.p), # Add sidelobes
            A.Lambda(
                name="Brightness perspective distortion",
                image=BrightnessGradient(
                    limits=(0.,1), primary_beam=True,
                    noise=0.01),
                p=self.p), # Gaussian Noise and pb brightness scaling
            # Modelling based transforms
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.1,
                rotate_limit=90, interpolation=2,
                border_mode=0, value=0,
                p=self.p),
            A.CenterCrop(
                width=85, height=85,
                p=1),
            A.Lambda(
                name='Dirty beam convlolution',
                image=Normalize(mean=0.5, std=0.5),
                always_apply=True)
        ]
        if self.aug_no is not None:
            augmentations = augmentations[:self.aug_no]
        if self.tensor_out:
            augmentations.append(ToTensorV2())
        augmentation = A.Compose(augmentations)
        return augmentation

    def default():
        return A.Compose([])


class Normalize():
    def __init__(self, mean=0.5, std=0.25):
        self.mean = mean
        self.std = std
    def __call__(self, image, **kwargs):
        transformed = image-image.min()
        transformed = transformed/transformed.max()
        transformed = transformed-self.mean
        transformed = transformed/self.std
        return transformed


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
            augmented_image = self.primary_beam_analogy(image)
            return augmented_image/augmented_image.max()
        else:
            augmented_image = self.gradient(image)
            return augmented_image

    def gradient(self, image):
        limits = self.limits[0] + np.diff(self.limits)[0]*np.sort(np.random.rand(2))
        grid = np.mgrid[0:image.shape[0], 0:image.shape[1]]
        normal = np.random.rand(2)*2-1
        scaling = (normal[0]*grid[0] - normal[1]*grid[1])
        scaling = scaling-scaling.min()+limits[0]
        transformed_image = image*scaling/scaling.max()*limits[1]

        return transformed_image

    def primary_beam_analogy(self, image):
        transform = A.Compose([
            A.GaussNoise(
                var_limit=self.noise*image.max(),
                mean=0, always_apply=True),
            A.Lambda(
                name="gradient",
                image=BrightnessGradient(
                    limits=self.limits,
                    primary_beam=False,
                    seed = self.seed,
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
        convolved_image = (np.abs(convolved_image)-np.abs(convolved_image).min())/np.abs(convolved_image).max()

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
