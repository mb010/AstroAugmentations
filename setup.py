import setuptools

setuptools.setup(
    name='AstroAugmentations',
    version='0.1.0',
    description='Augmentations for astronomical data sets',
    url='https://github.com/mb010/AstroAugmentations',
    project_urls={
        "Bug Tracker": "https://github.com/mb010/AstroAugmentations/issues",
    },
    author='Micah Bowles',
    author_email='micah.bowles@postgrad.manchester.ac.uk',
    license='MIT License',
    package_dir={"": "./"},
    packages=[
        "astroaugmentations",
        "astroaugmentations.datasets",
        "astroaugmentations.utils"
    ],
    install_requires=[
        'albumentations>=1.1.0',
        'numpy',
        'h5py>=3.6.0',
        'Pillow>=9.0.0',
        'scikit-image>=0.18.3',
        'scikit-learn>=1.0.2',
        'scipy>=1.7.3 '
    ],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Scientific/Engineering :: Image Processing'
    ],
)
