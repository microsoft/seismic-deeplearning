# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import logging
import os
import subprocess
import sys

from azure.common.client_factory import get_client_from_cli_profile
from azure.common.credentials import get_cli_profile
from azure.mgmt.resource import SubscriptionClient
from prompt_toolkit import prompt
from tabulate import tabulate
from toolz import pipe

from knack.util import CLIError

_GREEN = "\033[0;32m"
_BOLD = "\033[;1m"


def _run_az_cli_login():
    process = subprocess.Popen(["az", "login"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for c in iter(lambda: process.stdout.read(1), b""):
        sys.stdout.write(_GREEN + _BOLD + c.decode(sys.stdout.encoding))


def list_subscriptions(profile=None):
    """Lists the subscriptions associated with the profile
    
    If you don't supply a profile it will try to get the profile from your Azure CLI login

    Args:
        profile (azure.cli.core._profile.Profile, optional): Profile you wish to use. Defaults to None.
    
    Returns:
        list: list of subscriptions
    """
    if profile is None:
        profile = subscription_profile()
    cred, _, _ = profile.get_login_credentials()
    sub_client = SubscriptionClient(cred)
    return [
        {"Index": i, "Name": sub.display_name, "id": sub.subscription_id}
        for i, sub in enumerate(sub_client.subscriptions.list())
    ]


def subscription_profile():
    """Return the Azure CLI profile
    
    Returns:
        azure.cli.core._profile.Profile: Azure profile
    """
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
    print(f"You selected index {prompt_result} sub id {selected_sub['id']} name {selected_sub['Name']}")
    return selected_sub["id"]


def select_subscription(profile=None, sub_name_or_id=None):
    """Sets active subscription
    
    If you don't supply a profile it will try to get the profile from your Azure CLI login
    If you don't supply a subscription name or id it will list ones from your account and ask you to select one

    Args:
        profile (azure.cli.core._profile.Profile, optional): Profile you wish to use. Defaults to None.
        sub_name_or_id (str, optional): The subscription name or id to use. Defaults to None.

    Returns:
        azure.cli.core._profile.Profile: Azure profile
    
    Example:
    >>> profile = select_subscription()
    """
    if profile is None:
        profile = subscription_profile()

    if sub_name_or_id is None:
        sub_name_or_id = _prompt_sub_id_selection(profile)

    profile.set_active_subscription(sub_name_or_id)
    return profile
