from setuptools import setup, find_packages


def readme():
    with open("README.rst") as f:
        return f.read()


def exclude_list():
    return ["tests"]


setup(
    name="sct",
    version="1.0.1",
    author="ARESYS S.r.l.",
    author_email="info@aresys.it",
    description="SAR Calibration Tool",
    long_description=readme(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    license="MIT",
    packages=find_packages(exclude=exclude_list()),
)
