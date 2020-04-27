from setuptools import setup, Extension
from setuptools import find_packages

import yawml

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()


if __name__ == "__main__":
    setup(
        name="yawml",
        version=sandesh.__version__,
        description="YAWML: Yet Another Wrapper for Machine Learning",
        long_description=long_description,
        long_description_content_type='text/markdown',
        author="Abhishek Thakur",
        author_email="abhishek4@gmail.com",
        url="https://github.com/abhishekkrthakur/yawml",
        download_url="https://github.com/abhishekkrthakur/yawml/archive/v0.0.1.zip",
        license="MIT License",
        packages=find_packages(),
        include_package_data=True,
        platforms=["linux", "unix"],
        python_requires='>3.5.2'
    )
