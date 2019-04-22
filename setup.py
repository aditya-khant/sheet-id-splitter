import setuptools


setuptools.setup(
    name="measure_segmentation",
    version=1.0,
    install_requires=[
        "numpy",
        "matplotlib",
        "opencv-python",
        "scikit-image",
    ],
    packages=setuptools.find_packages(),
)
