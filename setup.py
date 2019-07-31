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
    description="DeepSeismic",
    install_requires=[],
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    name="deepseismic",
    packages=setuptools.find_packages(
        include=["deepseismic", "deepseismic.*"]
    ),
    platforms="any",
    python_requires=">= 3.5",
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    url="https://github.com/microsoft/deepseismic",
    version="0.0.1",
)
