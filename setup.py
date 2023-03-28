from setuptools import setup, find_packages


setup(
    name="datahub",
    description="ML datasets package",
    author="Gavin Sellers",
    license="MIT",
    platforms="any",
    package_dir={"": "datahub"},
    packages=find_packages("datahub"),
    install_requires=[
        "tensorflow-datasets>=4.8.3",
    ],
)
