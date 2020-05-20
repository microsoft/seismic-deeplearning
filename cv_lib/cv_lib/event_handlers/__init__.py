# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from ignite.handlers import ModelCheckpoint
import glob
import os
from shutil import copyfile


class SnapshotHandler:
    def __init__(self, dir_name, filename_prefix, score_function, snapshot_function):
        self._model_save_location = dir_name
        self._running_model_prefix = filename_prefix + "_running"
        self._snapshot_prefix = filename_prefix + "_snapshot"
        self._snapshot_function = snapshot_function
        self._snapshot_num = 1
        self._score_function = score_function
        self._checkpoint_handler = self._create_checkpoint_handler()

    def _create_checkpoint_handler(self):
        return ModelCheckpoint(
            self._model_save_location,
            self._running_model_prefix,
            score_function=self._score_function,
            n_saved=1,
            create_dir=True,
            save_as_state_dict=True,
            require_empty=False,
        )

    def __call__(self, engine, to_save):
        self._checkpoint_handler(engine, to_save)
        if self._snapshot_function():
            files = glob.glob(os.path.join(self._model_save_location, self._running_model_prefix + "*"))            
            name_postfix = os.path.basename(files[0]).lstrip(self._running_model_prefix)
            copyfile(
                files[0],
                os.path.join(self._model_save_location, f"{self._snapshot_prefix}{self._snapshot_num}{name_postfix}",),
            )
            self._checkpoint_handler = self._create_checkpoint_handler()  # Reset the checkpoint handler
            self._snapshot_num += 1
