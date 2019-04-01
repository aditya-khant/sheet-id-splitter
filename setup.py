import setuptools


setuptools.setup(
    name="score_splitter",
    version=0.1,
    install_requires=[
        "numpy",
        "matplotlib",
        "opencv-python",
        "scikit-image",
    ],
    packages=setuptools.find_packages(),
)
