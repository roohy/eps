import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pseudo-sampler-roohy",
    version="0.1.1",
    author="Ruhollah Shemirani",
    author_email="shemirani.r@gmail.com",
    description="Extreme Pseudo Sampling Package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/roohy/eps",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)