from typing import Any
import numpy as np
import albumentations as A
import torch

__all__ = [
    "MinMaxNormalize",
    "ToGray",
    "BrightnessGradient",
    "NaivePNGnorm",
    "ToTensor",
]


class ToTensor:
    def __init__(self, dtype=torch.float32) -> None:
        self.dtype = dtype

    def __call__(self, image, **kwds) -> Any:
        return torch.from_numpy(image).type(self.dtype)


class NaivePNGnorm:
    def __init__(self) -> None:
        pass

    def __call__(self, image, **kwargs):
        image = np.where(image >= 0, image, 0)
        image = image / np.amax(image)
        image = image * 255
        return image.astype(np.uint8)


class MinMaxNormalize:
    def __init__(self, mean=0.5, std=0.5, minmax=True):
        self.mean = mean
        self.std = std

    def __call__(self, image, **kwargs):
        transformed = image - image.min()
        max = 1.0 if transformed.max() == 0 else transformed.max()
        transformed = transformed / max
        transformed = transformed - self.mean
        transformed = transformed / self.std
        transformed = transformed - transformed.min()
        transformed = transformed / transformed.max()
        return transformed


class ToGray:
    """Makes a color image gray scale.
    Args:
        reduce_channels (bool):
            Decides if output has one channel (True) or three channels (False). Default: False
    """

    def __init__(self, reduce_channels=False):
        if reduce_channels:
            self.mean = lambda arr: arr.mean(axis=2, keepdims=True)
        else:
            self.mean = lambda arr: arr.mean(axis=2, keepdims=True).repeat(3, axis=2)

    def __call__(self, image, **kwargs):
        return self.mean(image)


class BrightnessGradient:
    """Randomly applies a linear gradient to the brightness of the image.
    Args:
        limits (tuple):
            Range to sample the lowest gradient from (peak is always 1).
            The plane is adjusted to make the minimum value of the plane
            equal the sampled value within the limits range. Default: [0.0,1.0]
        primary_beam (bool):
            Whether or not to emulate an incorrectly corrected primary beam.
        noise (float):
            Fraction of maximum signal at which the randomly selected
            gaussian noise should be capped.
        seed (float):
            Seed used for random numpy values.
    """

    def __init__(
        self, limits=[0.0, 1.0], primary_beam: bool = False, seed=None, noise=0.2
    ):
        self.seed = seed
        if self.seed is not None:
            np.random.RandomState.seed = self.seed
        self.noise = noise
        self.primary_beam = primary_beam
        self.limits = np.asarray(limits)
        self.rng = np.random.default_rng()

    def __call__(self, image, **kwargs):
        if self.primary_beam:
            augmented_image = self.primary_beam_analogy(image)
            max = 1.0 if augmented_image.max() == 0 else augmented_image.max()
            return augmented_image / max
        else:
            augmented_image = self.gradient(image)
            return augmented_image

    def gradient(self, image):
        limit = self.rng.uniform(
            *self.limits, 1
        )  # single random value in range of limits
        image_scaled = limit * image.copy()  # scale image to sampled value

        grid = np.mgrid[0 : image.shape[0], 0 : image.shape[1]]
        normal = self.rng.uniform(-1, 1, 2)

        scaling = normal[0] * grid[0] - normal[1] * grid[1]
        scaling = scaling - scaling.min() + limit
        max = 1.0 if scaling.max() == 0 else scaling.max()
        if len(image.shape) == 3:
            scaling = np.broadcast_to(scaling[:, :, np.newaxis], image.shape)
        transformed_image = image * scaling / max * self.limits[1]
        if image.dtype.name == "uint8":
            transformed_image = transformed_image.astype(np.uint8)

        return transformed_image

    def primary_beam_analogy(self, image):
        transform = A.Compose(
            [
                A.GaussNoise(
                    var_limit=self.noise * image.max(), mean=0, always_apply=True
                ),
                A.Lambda(
                    name="gradient",
                    image=BrightnessGradient(
                        limits=self.limits,
                        primary_beam=False,
                        seed=self.seed,
                        noise=self.noise,
                    ),
                    always_apply=True,
                ),
            ]
        )
        return transform(image=image)["image"]
