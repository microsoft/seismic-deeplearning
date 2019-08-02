import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    author="DeepSeismic Maintainers",
    author_email="deepseismic@microsoft.com",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
    ],
    dependency_links=[
        "https://github.com/opesci/devito/archive/v3.4.tar.gz#egg=devito-3.4"
    ],
    description="DeepSeismic",
    install_requires=[
        "devito==3.4",
        "h5py==2.9.0",
        "numpy==1.17.0",
        "scipy==1.3.0",
        "sympy==1.4",
    ],
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    name="deepseismic",
    packages=setuptools.find_packages(
        include=["deepseismic", "deepseismic.*"]
    ),
    platforms="any",
    python_requires=">= 3.5",
    scripts=["bin/vpgen"],
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    url="https://github.com/microsoft/deepseismic",
    version="0.0.1",
)
