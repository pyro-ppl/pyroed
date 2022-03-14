import re
import sys

from setuptools import find_packages, setup

with open("pyroed/__init__.py") as f:
    for line in f:
        match = re.match('^__version__ = "(.*)"$', line)
        if match:
            __version__ = match.group(1)
            break

try:
    long_description = open("README.md", encoding="utf-8").read()
except Exception as e:
    sys.stderr.write("Failed to read README.md: {}\n".format(e))
    sys.stderr.flush()
    long_description = ""

setup(
    name="pyroed",
    version=__version__,
    description="Sequence design using Pyro",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(include=["pyroed"]),
    package_data={"pyroed": ["py.typed"]},
    url="https://github.com/broadinstitute/pyroed",
    author="Pyro team at the Broad Institute of MIT and Harvard",
    author_email="fritz.obermeyer@gmail.com",
    install_requires=[
        "pyro-ppl>=1.7",
        "pandas",
    ],
    extras_require={
        "test": [
            "black",
            "isort>=5.0",
            "flake8",
            "pytest>=5.0",
            "mypy>=0.812",
        ],
    },
    python_requires=">=3.7",
    keywords="optimal experimental design pyro",
    license="Apache 2.0",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
