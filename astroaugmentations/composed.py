import numpy as np
import albumentations as A
from . import image_domain as image_domain
from albumentations.pytorch import ToTensorV2

class ComposedAugmentation():
    """Base class for composed augmentations.
    """
    def __init__(self, domain:str, tensor_out:bool=False):
        self.domain=domain
        self.tensor_out=False
        self.augmentations_list=[]
    def __str__(self):
        return f"ComposedAugmentation(domain:str={self.domain})"
    def _tensor_check(self):
        if self.tensor_out:
            self.augmentations_list.append(ToTensorV2())

class ImgRadio(ComposedAugmentation):
    def __init__(self, kernel:np.ndarray=None, p=0.5, tensor_out:bool=True, aug_no:int=None):
        super().__init__('radio', tensor_out=tensor_out)
        self.kernel = kernel
        self.aug_no = aug_no
        if type(p)==list:
            self.p = p
        else:
            self.p = [p for i in range(25)]
        self.augmentations_list = [
            # Change source perspective
            A.Lambda(
                name="Superpixel spectral index change",
                image=image_domain.radio.SpectralIndex(
                    mean=-0.8, std=0.2, super_pixels=True,
                    n_segments=100, seed=None),
                p=self.p[0]), # With segmentation
            A.Lambda(
                name="Brightness perspective distortion",
                image=image_domain.BrightnessGradient(limits=(0.,1.)),
                p=self.p[1]), # No noise
            A.ElasticTransform( # Elastically transform the source
                alpha=1, sigma=100, alpha_affine=25, interpolation=1,
                border_mode=1, value=0,
                p=self.p[2]),
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.1,
                rotate_limit=90, interpolation=2,
                border_mode=0, value=0,
                p=1),
            A.VerticalFlip(
                p=0.5),
            # Change properties of noise / imaging artefacts
            A.Lambda(
                name="Spectral index change of whole image",
                image=image_domain.radio.SpectralIndex(
                    mean=-0.8, std=0.2, seed=None),
                p=self.p[5]), # Across the whole image
            A.Emboss(
                alpha=(0.2,0.5), strength=(0.2,0.5),
                p=self.p[6]), # Quick emulation of incorrect w-kernels # Doesnt force the maxima to 1
            A.Lambda(
                name='Dirty beam convlolution',
                image=image_domain.radio.CustomKernelConvolution(
                    kernel=self.kernel, rfi_dropout=0.4, psf_radius=1.3,
                    sidelobe_scaling=1, mode='sum'),
                p=self.p[7]), # Add sidelobes
            A.Lambda(
                name="Brightness perspective distortion",
                image=image_domain.BrightnessGradient(
                    limits=(0.,1), primary_beam=True,
                    noise=0.01),
                p=self.p[8]), # Gaussian Noise and pb brightness scaling
            # Modelling based transforms
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.1,
                rotate_limit=90, interpolation=2,
                border_mode=0, value=0,
                p=self.p[9]),
            A.CenterCrop(
                width=85, height=85,
                p=1),
            A.Lambda(
                name='Min-Max Normalize',
                image=image_domain.MinMaxNormalize(mean=0.5, std=0.5),
                p=1)
        ]
        if self.aug_no is not None:
            self.augmentations_list = self.augmentations_list[:self.aug_no]
        self._tensor_check()
        self.augmentation = A.Compose(self.augmentations_list)

    def __call__(self, image, **kwargs):
        return self.augmentation(image=image)['image']

class ImgOptical(ComposedAugmentation):
    def __init__(self, dataset, p=0.5, tensor_out:bool=True, aug_no:int=None):
        super().__init__('optical', tensor_out=tensor_out)
        self.aug_no = aug_no
        template_transform = A.Compose([
            A.ToFloat(),
            A.ShiftScaleRotate(
                shift_limit=0.5,
                scale_limit=(0.8,1.2),
                rotate_limit=180,
                interpolation=1,
                border_mode=1,
                p=1
            ),
            A.VerticalFlip(p=0.5)
        ])
        if type(p)==list:
            self.p = p
        else:
            self.p = [p for i in range(10)]
        self.augmentations_list = [
            A.ToFloat(),
            # Slightly change source perspective
            A.ElasticTransform(
                alpha=1, sigma=100, alpha_affine=2, interpolation=1,
                border_mode=1, value=0,
                p=self.p[0]
            ),
            # Augment source confussion / overlap
            A.Lambda(
                name='AddSersicSources',
                image=image_domain.optical.SuperimposeSources(
                    mode='sersic',
                    max_number=5,
                    extent=(5,80),
                    scaling='default'
                ),
                p=self.p[1]
            ),
            A.Lambda(
                name='AddGaussianSources',
                image=image_domain.optical.SuperimposeSources(
                    mode='gaussian',
                    max_number=5,
                    extent=(3,20),
                    scaling=None
                ),
                p=self.p[2]
            ),
            # Change properties of noise / imaging artefacts
            A.Lambda(
                name='AddingRealData',
                image=image_domain.optical.CroppedTemplateOverlap(
                    mode='dataset', dataset=dataset,
                    template_transform=template_transform
                ),
                p=self.p[3]
            ),
            A.Lambda(
                name="Brightness perspective distortion",
                image=image_domain.BrightnessGradient(
                    limits=[0.5, 1]),
                p=self.p[4]
            ),
            # Model based transforms
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.1,
                rotate_limit=180, interpolation=2,
                border_mode=0,
            p=1),
            # Imaging artefacts
            A.Lambda(
                name="MissingData",
                image=image_domain.optical.ChannelWiseDropout(#
                    max_fraction=0.2,
                    min_width=10,
                    min_height=10,
                    max_holes=10,
                    channelwise_application=True
                ),
                p=self.p[6]
            )
        ]
        if self.aug_no is not None:
            self.augmentations_list = self.augmentations_list[:self.aug_no]
        self._tensor_check()
        self.augmentation = A.Compose(self.augmentations_list)

    def __call__(self, image, **kwargs):
        return self.augmentation(image=image)['image']
