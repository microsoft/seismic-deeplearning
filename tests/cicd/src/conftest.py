# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest


def pytest_addoption(parser):
    parser.addoption("--nbname", action="store", type=str, default=None)
    parser.addoption("--dataset_root", action="store", type=str, default=None)
    parser.addoption("--model_pretrained", action="store", type=str, default=None)
    parser.addoption("--cwd", action="store", type=str, default="examples/interpretation/notebooks")


@pytest.fixture
def nbname(request):
    return request.config.getoption("--nbname")


@pytest.fixture
def dataset_root(request):
    return request.config.getoption("--dataset_root")

@pytest.fixture
def model_pretrained(request):
    return request.config.getoption("--model_pretrained")

@pytest.fixture
def cwd(request):
    return request.config.getoption("--cwd")

"""
def pytest_generate_tests(metafunc):
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".
    option_value = metafunc.config.option.nbname
    if 'nbname' in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("nbname", [option_value])
    option_value = metafunc.config.option.dataset_root
    if 'dataset_root' in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("dataset_root", [option_value])
"""
