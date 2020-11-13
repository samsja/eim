import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()



with open("requirements.txt") as f:
    packages = [line.strip() for line in f.readlines()]


setuptools.setup(
    name="eim-samsja-faycal", # Replace with your own username  
    version="0.0.1",
    author="Sami Jaghouar And Faycal Rekbi",
    author_email="sami.jaghouar@hotmail.fr faycal.rekbdi@etu.utc.fr",
    description="implementation of Empirical Interpolation Methods (EIM) algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/samsja/eim",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires = packages
)
