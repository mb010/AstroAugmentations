import setuptools

setuptools.setup(
    name='AstroAugmentations',
    version='0.1.0',
    description='Augmentations for astronomical data sets',
    url='https://github.com/mb010/AstroAugmentations',
    author='Micah Bowles',
    author_email='micah.bowles@postgrad.manchester.ac.uk',
    license='MIT License',
    package_dir={"": "AstroAugmentations"},
    packages=setuptools.find_packages(where="AstroAugmentations"),
    install_requires=[
        'albumentations>=1.1.0',
        'numpy',
        'h5py>=3.6.0',
        'Pillow>=9.0.0',
        'scikit-image>=0.18.3',
        'scikit-learn>=1.0.2',
        'scipy>=1.7.3 '
        #'torch>=1.10.2+cu113',
        #'torchvision>=0.11.3+cu113'
    ],
    # torch and torchvision are also required!
    # Which version you install depends on your system: https://pytorch.org/

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
