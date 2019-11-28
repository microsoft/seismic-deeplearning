# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

def pytest_addoption(parser):
    parser.addoption("--nbname", action="store", type=str, default=None)

def pytest_generate_tests(metafunc):
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".
    option_value = metafunc.config.option.nbname
    if 'nbname' in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("nbname", [option_value])
