"""
Setup script for OPTICUS
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="opticus",
    version="0.1.0",
    author="IceCube Collaboration",
    description="Optical Property Transformer for IceCube Upgrade Camera System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10.0",
        "torchvision>=0.11.0",
        "numpy>=1.20.0",
        "h5py>=3.0.0",
        "matplotlib>=3.3.0",
        "PyYAML>=5.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.0",
            "flake8>=3.9.0",
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "opticus-train=opticus.scripts.train:main",
            "opticus-eval=opticus.scripts.evaluate:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    include_package_data=True,
    package_data={
        "opticus": ["configs/*.yaml"],
    },
)

