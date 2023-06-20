from setuptools import find_packages, setup

setup(
    name="colorizer",
    version="1.0",
    packages=find_packages(),
    package_data={
        "colorizer": ["models/*"],
    },
    author="Daniel Stoffel",
    author_email="daniel.stoffel@stud.th-deg.de",
    description="This package is for colorizing black and white images using machine learning. This is a project for the Computer Vision Course in the 4. Semester of the AI Program from the TH Deggendorf.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/CRY-TeX/cv-black-white-images-to-color",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    install_requires=[
        "numpy",
        "seaborn",
        "scikit-learn",
        "scikit-image",
        "tqdm",
        "tensorflow-cpu"
    ],
)
