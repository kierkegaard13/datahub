from setuptools import setup, find_packages


setup(
    name="datahub",
    description="ML datasets package",
    author="Gavin Sellers",
    license="MIT",
    platforms="any",
    packages=find_packages(),
    install_requires=[
        "tensorflow-datasets>=4.8.3",
    ],
)
