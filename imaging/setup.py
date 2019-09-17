import setuptools

with open("../README.md", "r") as f:
    long_description = f.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

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
        "https://github.com/opesci/devito/archive/v3.5.tar.gz#egg=devito-3.5"
    ],
    description="DeepSeismic",
    install_requires=requirements,
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    name="deepseismic_imaging",
    packages=setuptools.find_packages(
        include=["deepseismic_imaging", "deepseismic_imaging.*"]
    ),
    platforms="any",
    python_requires=">= 3.6",
    scripts=["bin/ds"],
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    url="https://github.com/microsoft/deepseismic",
    version="0.1.0",
    zip_safe=False,
)
