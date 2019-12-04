# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
import os
import subprocess
import sys

from azure.common.client_factory import get_client_from_cli_profile
from azure.common.credentials import get_cli_profile
from azure.mgmt.resource import SubscriptionClient
from knack.util import CLIError

# Importing here to avoid overhead to the rest of the application
from tabulate import tabulate
from toolz import pipe
import json
from prompt_toolkit import prompt

_GREEN = "\033[0;32m"
_BOLD = "\033[;1m"


def _run_az_cli_login():
    process = subprocess.Popen(
        ["az", "login"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    for c in iter(lambda: process.stdout.read(1), b""):
        sys.stdout.write(_GREEN + _BOLD + c.decode(sys.stdout.encoding))


def list_subscriptions(profile=None):
    if profile is None:
        profile = subscription_profile()
    cred, _, _ = profile.get_login_credentials()
    sub_client = SubscriptionClient(cred)
    return [
        {"Index": i, "Name": sub.display_name, "id": sub.subscription_id}
        for i, sub in enumerate(sub_client.subscriptions.list())
    ]


def subscription_profile():
    logger = logging.getLogger(__name__)
    try:
        return get_cli_profile()
    except CLIError:
        logger.info("Not logged in, running az login")
        _run_az_cli_login()
        return get_cli_profile()


def _prompt_sub_id_selection(profile):
    sub_list = list_subscriptions(profile=profile)
    pipe(sub_list, tabulate, print)
    prompt_result = prompt("Please type in index of subscription you want to use: ")
    selected_sub = sub_list[int(prompt_result)]
    print(
        f"You selected index {prompt_result} sub id {selected_sub['id']} name {selected_sub['Name']}"
    )
    return selected_sub["id"]


def select_subscription(profile=None, sub_name_or_id=None):
    if profile is None:
        profile = subscription_profile()

    if sub_name_or_id is None:
        sub_name_or_id = _prompt_sub_id_selection(profile)

    profile.set_active_subscription(sub_name_or_id)
    return profile


if __name__ == "__main__":
    profile = select_subscription()
    print("here")
