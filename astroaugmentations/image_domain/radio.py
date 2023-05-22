import numpy as np
import albumentations as A
from skimage.segmentation import slic
from typing import Optional, Any, List

__all__ = ["SpectralIndex", "CustomKernelConvolution", "RFI", "UVAugmentation"]


class RFI:
    """Emulates RFI in interferometers by adding random brightness points into the fourrier space of the respective image.

    Args:
        var_limit (float): The variance of the gaussian from which the RFI is sampled.
        rfi_prob (Optional[float], optional): Probability of a given pixel to have RFI added. Defaults to 0.01.
        per_channel (Optional[bool], optional): Apply on a per channel bases if True. Defaults to True.
        seed (Optional[int], optional): RFI generation seeding. Does not seed where the RFI is, just the normal samples used. Defaults to None.
        fft (bool, optional): Transform the image to fourrier space within this transform, or not (if image is passed in fourrier space already). Defaults to False.
    """

    def __init__(
        self,
        var_limit: float,
        rfi_prob: Optional[float] = 0.01,
        per_channel: Optional[bool] = True,
        seed: Optional[int] = None,
        fft: bool = True,
    ) -> None:
        self.rng = np.random.default_rng(seed=seed)
        self.var = self.rng.random() * abs(var_limit)
        self.fft = fft
        self._reverse_PixelDropout = A.PixelDropout(
            dropout_prob=1 - rfi_prob, per_channel=per_channel, always_apply=True
        )

    def __call__(self, image, **kwargs) -> Any:
        if self.fft:
            image = np.fft.fft2(image)
        rfi = self.rng.normal(
            loc=0.0, scale=self.var, size=[2] + [v for v in image.shape]
        )
        rfi = rfi[0] + 1j * rfi[1]
        rfi = self._reverse_PixelDropout(image=rfi)["image"]
        image = image * (rfi + 1)
        if self.fft:
            return np.real(np.fft.ifft2(image))
        return image


class UVNoise:
    """Emulates antenna based noise."""

    def __init__(
        self,
        var_limit: float,
        mean: float = 0.0,
        rfi_prob: Optional[float] = 0.01,
        per_channel: Optional[bool] = True,
        seed: Optional[int] = None,
        fft: bool = True,
    ) -> None:
        self.rng = np.random.default_rng(seed=seed)
        self.var = self.rng.random() * abs(var_limit)
        self.fft = fft

    def __call__(self, image, **kwargs) -> Any:
        if self.fft:
            image = np.fft.fft2(image)
        rfi = self.rng.normal(
            loc=0.0, scale=self.var, size=[2] + [v for v in image.shape]
        )
        rfi = rfi[0] + 1j * rfi[1]
        image = image * (rfi + 1)
        if self.fft:
            return np.real(np.fft.ifft2(image))
        return image


class UVAugmentation:
    # Band pass calibration errors
    # Antenna pointing errors
    # Phase calibration errors (per antenna)
    # Clean errors - over cleaning - negative features
    # Self-cal erros
    def __init__(
        self,
        dropout_p: Optional[float] = None,
        dropout_mag: Optional[float] = None,
        noise_p: Optional[float] = None,
        noise_mag: Optional[float] = None,
        rfi_p: Optional[float] = None,
        rfi_mag: Optional[float] = None,
        rfi_prob: Optional[float] = None,
        fft: bool = False,
        out: Optional[List[callable]] = None,
    ) -> None:
        """Augments an image in the fourrier space emulating additional RFI flagging, noise, and RFI.

        Args:
            dropout_p (Optional[float], optional): Probability that RFI flagging (pixelwise dropout in fourrier space) is applied. Defaults to None.
            dropout_mag (Optional[float], optional): Probability that a given pixel is dropped in fourrier space. Defaults to None.
            noise_p (Optional[float], optional): Probability that additional noise is added. Defaults to None.
            noise_mag (Optional[float], optional): Upper variance range to sample from to generate Gaussian noise. Defaults to None.
            rfi_p (Optional[float], optional): Probability that RFI augmentation is applied. Defaults to None.
            rfi_mag (Optional[float], optional): Variance of normal distribution for RFI sampling. Defaults to None.
            rfi_prob (Optional[float], optional): Probability of a given pixel to have RFI added. Defaults to None.
            fft (bool): Parameter to return image in fourrier space (True). Defaults to False.
        """
        self.out = out
        transforms = []
        if (dropout_mag is not None) and (dropout_p is not None):
            transforms.append(
                # Pixelwise dropout (RFI flagging)
                A.PixelDropout(
                    dropout_prob=dropout_mag,
                    p=dropout_p,
                    per_channel=True,
                )
            )
        if (noise_mag is not None) and (noise_p is not None):
            # Add gaussian noise
            # Do I have to manually write a gaussian noise for complex numbers?
            transforms.append(
                A.Lambda(
                    name="UVNoise",
                    image=UVNoise(
                        var_limit=noise_mag,
                        mean=0,
                        per_channel=True,
                        fft=False,
                    ),
                    p=noise_p,
                )
            )
        if (rfi_mag is not None) and (rfi_p is not None) and (rfi_prob is not None):
            # Add agressive antenae noise
            transforms.append(
                A.Lambda(
                    name="RFI injection",
                    image=RFI(
                        var_limit=rfi_mag,
                        rfi_prob=rfi_prob,
                        per_channel=False,
                        seed=None,
                        fft=False,
                    ),
                    p=rfi_p,
                )
            )
        self.transform = A.Compose(transforms)
        self.fft = fft

    def __call__(self, image, **kwargs) -> Any:
        image = np.nan_to_num(image, nan=0)
        max_ = np.max(np.abs(image))
        max_ = max_ if max_ != 0 else 1
        uv = np.fft.fft2(image / max_)
        uv = self.transform(image=uv)["image"]
        if not self.fft:
            uv = np.real(np.fft.ifft2(uv))
        if self.out is not None:
            uv = np.stack([o(uv) for o in self.out])
        return uv


class SpectralIndex:
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

    def __init__(
        self,
        mean=-0.8,
        std=0.2,
        super_pixels=False,
        n_segments=225,
        seed=None,
        compactness=0.0001,
        segmentation_out=False,
    ):
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
                    segments == seg_val, self.random_spectral_scaling(), segment_scaling
                )
            augmented_image = augmented_image * segment_scaling
        else:
            augmented_image = augmented_image * self.random_spectral_scaling()
        if self.segmentation_out:
            return augmented_image, segments
        else:
            return augmented_image

    def segment(self, image):
        segments = slic(
            image,
            n_segments=self.n_segments,
            slic_zero=True,
            compactness=self.compactness,
            start_label=1,
        )
        return segments

    def random_spectral_scaling(self):
        sample = np.random.randn(1)[0] * self.std + self.mean
        pct_brightness_change = (self.mean - sample) / self.mean
        return 1.0 + pct_brightness_change


class CustomKernelConvolution:
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

    def __init__(
        self,
        kernel,
        rfi_dropout=None,
        rfi_dropout_p=1,
        beam_cut=None,
        psf_radius=None,
        sidelobe_scaling=1,
        show_kernel=False,
        mode="sum",
    ):
        self.mode = mode
        assert self.mode in ["masked", "sum", "delta"]
        self.kernel = kernel
        self.rfi_dropout = rfi_dropout
        self.rfi_dropout_p = rfi_dropout_p
        self.beam_cut = beam_cut
        self.psf_radius = psf_radius
        self.show_kernel = show_kernel
        self.sidelobe_scaling = sidelobe_scaling

    def __call__(self, image, **kwargs):
        # Match kernel size to image size
        kernel = np.zeros(image.shape, dtype=np.complex128)
        im_mid = [image.shape[0] // 2, image.shape[1] // 2]
        k_mid = [self.kernel.shape[0] // 2, self.kernel.shape[1] // 2]
        kernel[
            im_mid[0] - k_mid[0] : im_mid[0] + k_mid[0],
            im_mid[0] - k_mid[0] : im_mid[0] + k_mid[0],
        ] = self.kernel

        # Adjust sidelobes to match expected noise level
        # Set central beam above self.beam_cut to 0
        if self.beam_cut is not None:
            kernel = self.beam_cutting(kernel)
        # Remove central pixels if within given pixel radius
        kernel = self.radius_clipping(kernel) if self.psf_radius is not None else kernel

        # Add delta function to sample original image
        if self.mode == "delta":
            kernel[kernel.shape[0] // 2, kernel.shape[1] // 2] = (
                np.real(kernel).max() / self.max_sidelobe + 1j * 0
            )

        # FTs
        ft_img = np.fft.fft2(image)
        ft_kernel = np.fft.fft2(kernel)

        # RFI Flagging:
        if self.rfi_dropout is not None:
            transform = A.Compose(
                [A.PixelDropout(dropout_prob=self.rfi_dropout, p=self.rfi_dropout_p)]
            )
            ft_kernel = transform(image=ft_kernel)["image"]

        # Convolved image
        convolved_image = np.fft.fftshift(np.fft.ifft2(ft_img * ft_kernel))
        max = (
            1.0 if np.abs(convolved_image).max() == 0 else np.abs(convolved_image).max()
        )
        convolved_image = (
            np.abs(convolved_image) - np.abs(convolved_image).min()
        ) / max

        # Add back in original signal
        if self.mode == "masked":
            convolved_image = np.where(image > 0, 0, np.abs(convolved_image))
            convolved_image = convolved_image * self.sidelobe_scaling + image
        elif self.mode == "sum":
            covolved_image = (np.abs(convolved_image) + image) / 2
        return convolved_image

    def beam_cutting(self, kernel):
        kernel = np.where(
            np.abs(kernel) / np.abs(kernel).max() > self.beam_cut, 0, kernel
        )
        return kernel

    def radius_clipping(self, kernel):
        xx, yy = (
            np.mgrid[0 : kernel.shape[0], 0 : kernel.shape[0]] - kernel.shape[0] // 2
        )
        rr = np.sqrt(xx**2 + yy**2)
        kernel = np.where(rr <= self.psf_radius, 0, kernel)
        return kernel
