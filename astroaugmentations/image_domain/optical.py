import numpy as np
import albumentations as A
from astropy.modeling.models import Sersic2D
from cv2 import resize

__all__ = [
    "ChannelWiseDropout",
    "SuperimposeSources",
    "CroppedTemplateOverlap"
]

class ChannelWiseDropout():
    """Dropout random rectangles in a channelwise fashion up to a fraction of a total image.
    Args:
        max_fraction (float or list of floats, default 0.2):
            Maximum fraction of data which can be removed (true drop rate is sampled from range [0,max_fraction]).
            If a list is passed, its length must match the number of channels.
            If a list is used, each channel fractional drop out will be sampled according to the respective value in the list (if channelwise_application is True).
            If dropout is not being applied channelwise, then the maximum value passed in the list is used as the upper limit.
        fill_value (float, default 0):
            Value to fill the holes with.
        min_width (int):
            Minmimum width of holes to drop.
        min_height (int):
            Minmimum height of holes to drop.
        channelwise_application (bool, default False):
            Whether or not to apply dropouts on each channel (True) or to the whole image (False).
        max_holes (int):
            Limit the number of dropouts to apply to limit potential processing time.
        seed:
             Seed for the random processes.
    """
    def __init__(self,
                 max_fraction=0.2,
                 fill_value=0,
                 min_width=5,
                 min_height=5,
                 seed=None,
                 channelwise_application=False,
                 max_holes=100
                ):
        self.seed = seed
        if self.seed is not None:
            np.random.RandomState.seed = self.seed
        self.rng = np.random.default_rng()
        self.max_fraction = max_fraction
        self.min_height = min_height
        self.min_width = min_width
        self.fill_value = fill_value
        self.max_holes = max_holes
        self.channelwise_application = channelwise_application
        self.channelwise_fraction = True if type(self.max_fraction) not in [float, int] else False

    def __call__(self, image, **kwargs):

        self.fraction = self.rng.uniform(0, self.max_fraction)
        h, w, c = image.shape
        holes = 0
        if self.channelwise_application:
            if self.channelwise_fraction:
                self.pixels = h*w*self.fraction
                pixels_to_drop = self.rng.integers(self.min_height*self.min_width, [max(pix, self.min_height*self.min_width+1) for pix in self.pixels])
                for channel in range(0, image.shape[-1]):
                    holes=0
                    while (self._check_removed(image, channel=channel) and holes<self.max_holes):
                        image = self._channelwise_dropout(image, pixels_to_drop[channel], channel)
                        holes+=1
            else:
                self.pixels = h*w*c*self.fraction
                pixels_to_drop = self.rng.integers(self.min_height*self.min_width, max(self.pixels, self.min_height*self.min_width*c+1))
                while (self._check_removed(image) and holes<self.max_holes):
                    channel = self.rng.integers(0, image.shape[-1])
                    image = self._channelwise_dropout(image, pixels_to_drop, channel)
        else:
            self.pixels = int(h*w*max(self.fraction)) if type(self.fraction) in [list, np.ndarray] else int(h*w*self.fraction)
            pixels_to_drop = self.rng.integers(self.min_height*self.min_width, max(self.pixels, self.min_height*self.min_width*c+1))
            while (self._check_removed(image[:,:,0]) and holes<self.max_holes):
                image = self._dropout(image, pixels_to_drop)
                holes+=1

        return image

    def _dropout(self, image, pixels_to_drop):
        max_width = min(pixels_to_drop, image.shape[0])
        ratio  = self.rng.uniform(0,1)
        width  = int(ratio*pixels_to_drop**0.5)
        height = int((1-ratio)*pixels_to_drop**0.5)
        #height = max(pixels_to_drop//width, self.min_height)
        x0     = self.rng.integers(0, max(image.shape[0]-width, 1))
        y0     = self.rng.integers(0, max(image.shape[1]-height, 1))
        image[x0:x0+width, y0:y0+height] = self.fill_value
        return image

    def _channelwise_dropout(self, image, pixels_to_drop, channel):
        """Returns the image after a channelwise dropout is applied."""
        image[:,:,channel] = self._dropout(image[:,:,channel], pixels_to_drop)
        return image

    def _check_removed(self, image, channel=None):
        """Verifies if the dropout has reached the appropriate total of dropped pixels """
        minimum_area = self.min_width*self.min_height
        if channel is None:
            img = image
            pix = self.pixels
        else:
            img = image[:,:,channel]
            pix = self.pixels[channel]

        if np.where(img==0,1,0).sum() > pix-minimum_area:
            return False
        return True

class SuperimposeSources():
    """Adding other 2d gaussians or real data to the image to
    emulate other galaxies being in the field.
    Args:
        mode (str, default 'gaussians'):
            Though which mode to add sources. Default: 'gaussians'.
            Currently no other modes implemented.
        sigma_lims (tuple, default (0.5, 5.5)):
            Limits from which to sample the standard deviations of the gaussians.
        max_brigtness (float, default 1.):
            Upper limit of the brightness of the generated gaussians.
        max_number (int, default 4):
            Upper limit of how many sources are allowed to be added. Minimum is always 1.
    """
    def __init__(self,
                 mode:str='sersic',
                 extent=(20, 80),
                 max_brightness=1,
                 max_number=4,
                 seed=None,
                 sersic_indicies=(1,5), #(0.5,10)
                 scaling='default'
                ):
        self.seed = seed
        if self.seed is not None:
            np.random.RandomState.seed = self.seed
        self.mode = mode
        self.extent = extent
        self.max_brightness = max_brightness
        self.max_number = max_number
        self.rng = np.random.default_rng()#
        self.sersic_indicies = sersic_indicies
        self.scaling = scaling

    def __call__(self, image, **kwargs):
        if self.mode == 'gaussian':
            augmented_image = self._gaussians(image)
        elif self.mode == 'sersic':
            augmented_image = self._sersics(image)
        else:
            augmented_image = image
        return augmented_image

    def _gaussians(self, image):
        if image.dtype.name == 'uint8':
            image = A.ToFloat(always_apply=True)(image=image)['image']
        image_max = image.max()
        number = self.rng.integers(1, self.max_number+1) # Randomly select number of gaussians to place
        x0, y0 = self.rng.uniform(0.2, 1.2, (2, number)) # Randomly select centre coordinates in image
        x0 *= image.shape[0]
        y0 *= image.shape[1]
        sigx, sigy = self.rng.uniform(*self.extent, (2,number)) # Generate minor and major axes
        theta = self.rng.uniform(0, np.pi, (1, number)) # Generate random orientation
        amplitude = self.rng.uniform(0, self.max_brightness, (1, number)) # Generate random orientation

        grid = self._get_grid(image, number)

        # Generate arrays of parameterised Gaussians
        a = 0.5*np.cos(theta)**2/sigx**2 + 0.5*np.sin(theta)**2/sigy**2
        b = -0.25 * np.sin(2*theta)/sigx**2 + 0.25*np.sin(2*theta)/sigy**2
        c = 0.5*np.sin(theta)**2/sigx**2 + 0.5*np.cos(theta)**2/sigy**2
        gaussian = amplitude * np.exp(
            - (a*(grid[0]-x0)**2
            + 2*b*(grid[0]-x0)*(grid[1]-y0)
            + c*(grid[1]-y0)**2)
        )

        # If channelised image, sample colors from :
        if image.shape[-1]>1:
            gaussian_composite = []
            for i in range(number):
                gauss = np.broadcast_to(gaussian[:,:,i:i+1], image.shape).copy()
                gauss = self._scale(gauss)
                gauss = self._colorize_sample(gauss)
                gaussian_composite.append(gauss[np.newaxis])
            gaussian = np.sum(np.vstack(gaussian_composite), axis=0).astype(np.float32)
            image = A.TemplateTransform(gaussian.squeeze(), img_weight=1, always_apply=True)(image=image.astype(np.float32))['image']
        else:
            image = A.TemplateTransform(gaussian.squeeze(), img_weight=1, always_apply=True)(image=image.astype(np.float32))['image']
        image = self._safe_rescale(image, image_max)
        return image

    def _sersics(self, image):
        image_max = image.max()
        number = self.rng.integers(1, self.max_number+1) # Randomly select number of gaussians to place
        amplitude = self.rng.uniform(0, self.max_brightness, number) # Generate random orientation
        x0, y0 = self.rng.uniform(0.2, 1.2, (2, number)) # Randomly select centre coordinates in image
        x0   *= image.shape[0]
        y0   *= image.shape[1]
        r_eff = self.rng.uniform(*self.extent, number) # Generate an effective radius
        ellip = self.rng.uniform(0.3, 0.7, number) # sample semi-major : semi-major ratio
        theta = self.rng.uniform(0, np.pi, number) # Generate random orientation
        n     = self.rng.uniform(*self.sersic_indicies, number) # Sample Sersic index from physically inspired range

        grid = self._get_grid(image, number)

        sersics = []
        for i in range(number):
            model = Sersic2D(
                amplitude = 1,
                r_eff=r_eff[i],
                n=n[i], # Sersic index: n\in[0.5,10]
                x_0=x0[i],
                y_0=y0[i],
                ellip=ellip[i],
                theta=theta[i]
            )
            out = model(*grid[:,:,:,i])*amplitude[i]
            sersics.append(out)

        sersics = np.asarray(sersics).transpose(1,2,0)
        sersics = sersics/sersics.max() if sersics.max()!=0 else sersics

        if image.shape[-1]>1:
            sersics_composite = []
            for i in range(number):
                sersic = np.broadcast_to(sersics[:,:,i:i+1], image.shape).copy()
                sersic = self._colorize_sample(sersic)
                sersic = self._scale(sersic)
                sersics_composite.append(sersic[np.newaxis])
            sersics = np.sum(np.vstack(sersics_composite), axis=0).astype(np.float32)
            image = A.TemplateTransform(sersics.squeeze(), img_weight=1, always_apply=True)(image=image.astype(np.float32))['image']
        else:
            image = A.TemplateTransform(sersics.squeeze(), img_weight=1, always_apply=True)(image=image.astype(np.float32))['image']
        image = self._safe_rescale(image, image_max)
        return image

    def _get_grid(self, image, number):
        # Generate Gaussian image with same channel number as image
        grid = np.mgrid[0:image.shape[0]:1, 0:image.shape[1]:1]
        grid = grid[:,:,:, np.newaxis]
        grid = np.broadcast_to(grid, [*grid.shape[:3], number])
        return grid

    def _colorize_sample(self, sample):
        # Adjust colors to be in line with realistic galaxies
        # Sampling distributions estimated from central 20x20 pixels of the GalaxyMNIST training set:
        # https://github.com/mwalmsley/galaxy_mnist
        factors = np.asarray([1, self.rng.normal(1.01, 0.03), self.rng.normal(1.02, 0.04)])
        factors = factors/factors.max()
        sample = np.multiply(sample, factors[np.newaxis, np.newaxis, :])
        return sample

    def _safe_rescale(self, data, original_max_value=1):
        tmp_max = 1 if data.max() == 0 else data.max()
        return data/tmp_max*original_max_value

    def _scale(self, image):
        if self.scaling == 'default':
            return np.arcsinh(image)
        elif self.scaling is not None:
            return self.scaling(image)
        else:
            return image

class CroppedTemplateOverlap():
    """Overlaps image with other template.

    Randomly transforms image to have a template or randomly sampled
    data point to be superimposed onto the input image.

    Attributes:
        mode (str):
            Which mode the transformation works in. Either: 'template' or 'dataset' or 'catalog'
        template (np.ndarray)
        img_weight (float)
        dataset (indexable np.ndarrays with output for `len` function)
        template_transform (Compsed albumentations transfromation)
        catalog (pd.DataFrame) - catalog object for GZ datasets from https://github.com/mwalmsley/galaxy-datasets for superimposing images
        resolution (int) - resolution to resize galaxy-datasets catalog images to be pasted
    """
    def __init__(self,
                 mode:str='template',
                 template:np.ndarray=None,
                 img_weight:float=1,
                 dataset=None,
                 template_transform=A.Compose([A.ToFloat(p=1)]),
                 catalog = None,
                 resolution = None,
                ):
        """Init CroppedTemplateOverlap with given attributes."""
        self.template = template
        self.img_weight = img_weight
        self.dataset = dataset
        self.mode=mode
        self.template_transform = template_transform
        self.rng = np.random.default_rng()
        self.catalog = catalog
        self.resolution = resolution

    def __call__(self, image, **kwargs):
        """Apply generated transform to an input image."""
        if image.dtype == np.uint8:
            image = A.ToFloat(always_apply=True)(image=image)['image']
        if self.mode == 'template':
            height, width = image.shape[:2]
            sampled_template = A.Compose([
                A.ShiftScaleRotate(shift_limit=0., scale_limit=2, rotate_limit=180, interpolation=1, border_mode=0, value=0, always_apply=True),
                A.RandomCrop(height, width, always_apply=True),
            ])(image=self.template)['image']
            image = A.TemplateTransform(sampled_template, img_weight=1, always_apply=True)(image=image)['image']

        elif self.mode == 'dataset':
            height, width = image.shape[:2]
            datasample, _ = self.dataset[self.rng.integers(0, len(self.dataset))]
            #datasample = A.ToFloat(always_apply=True)(image=datasample)['image']
            datasample = self.template_transform(image=datasample)['image']
            image = A.TemplateTransform(datasample, img_weight=1, always_apply=True)(image=image)['image']
        
        elif self.mode == "catalog":
            try:
                from simplejpeg import decode_jpeg
            except ImportError:
                raise ImportError(
                    "simplejpeg is not installed. Install via https://pypi.org/project/simplejpeg/"
                )
            height, width = image.shape[:2]
            galaxy = self.catalog.iloc[self.rng.integers(0, len(self.catalog))]
            with open(galaxy["file_loc"], "rb") as f:
                datasample = decode_jpeg(f.read())
            datasample = resize(datasample, (self.resolution, self.resolution))
            datasample = self.template_transform(image=datasample)["image"]
            ShiftScaleRotate = A.ShiftScaleRotate(
                shift_limit=0.3,
                scale_limit=(-0.3, 0.3),
                rotate_limit=360,
                border_mode=3,
            )
            datasample = ShiftScaleRotate(image=datasample)["image"]
            image = A.TemplateTransform(datasample, img_weight=1, always_apply=True)(
                image=image
            )["image"]
        else:
            raise NotImplementedError
        return image
