# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
import papermill as pm

from tests.notebooks_common import OUTPUT_NOTEBOOK, KERNEL_NAME

# don't add any markup as this just runs any notebook which name is supplied
# @pytest.mark.integration
# @pytest.mark.notebooks
def test_notebook_run(nbname, dataset_root, model_pretrained, cwd):
    pm.execute_notebook(
        nbname,
        OUTPUT_NOTEBOOK,
        kernel_name=KERNEL_NAME,
        parameters={
            "max_epochs": 1,
            "max_snapshots": 1,
            "papermill": True,
            "dataset_root": dataset_root,
            "model_pretrained": model_pretrained,
        },
        cwd=cwd,
    )
