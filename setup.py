from setuptools import setup, find_packages

name = "geomechinterp"
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name=name,
    version="0.0.1",
    url="https://github.com/culpritgene/GeoMechInterp",
    author="Culpritgene",
    author_email="culpritgene@gmail.com",
    description="Geometric Interpretability of LLMs",
    packages=find_packages(exclude="tests"),
    install_requires=requirements,
    setup_requires=[
        "pytest-runner",
    ],
    tests_require=[
        "pytest",
    ],
)
