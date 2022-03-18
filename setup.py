import setuptools

setuptools.setup(
    name='AstroAugmentations',
    version='0.1.0',
    description='Augmentations for astronomical data sets.',
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
        "astroaugmentations.utils",
        "astroaugmentations.image_domain"
    ],
    install_requires=[
        'h5py',
        'astropy'
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
