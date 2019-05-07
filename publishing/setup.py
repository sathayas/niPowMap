import setuptools

with open("README.md", "r", encoding='UTF-8', newline='') as fh:
    long_description = fh.read()

setuptools.setup(
    name="publishing",
    version="0.0.1",
    author="Yeji Charlotte Yun",
    author_email="yeji.yun1225@utexas.edu",
    description="Python implementation of MatLab PowerMap software for neuroimaging studies; Directed by Dr. Satoru Hayasaka",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sathayas/niPowMap",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
          'nibabel', 'numpy', 'scipy', 'matplotlib', 'skimage'
      ]
)