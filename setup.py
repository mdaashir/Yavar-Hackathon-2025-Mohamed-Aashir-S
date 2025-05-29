from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="invoice-data-extractor",
    version="1.0.0",
    author="Mohamed Aashir S",
    author_email="s.mohamedaashir@gmail.com",
    description="A robust system for extracting and validating data from invoice PDFs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mdaashir/Yavar-Hackathon-2025-Mohamed-Aashir-S",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Office/Business :: Financial",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "invoice-extractor=src.main:main",
        ],
    },
    package_data={
        "": ["*.json", "*.yaml"],
    },
    include_package_data=True,
)
