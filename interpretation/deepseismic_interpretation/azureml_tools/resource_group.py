# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from azure.mgmt.resource import ResourceManagementClient
from azure.common.credentials import get_cli_profile
import logging


def _get_resource_group_client(profile_credentials, subscription_id):
    return ResourceManagementClient(profile_credentials, subscription_id)


def resource_group_exists(resource_group_name, resource_group_client=None):
    return resource_group_client.resource_groups.check_existence(resource_group_name)


class ResourceGroupException(Exception):
    pass


def create_resource_group(profile_credentials, subscription_id, location, resource_group_name):
    """Creates resource group if it doesn't exist
    
    Args:
        profile_credentials : credentials from Azure login 
        subscription_id (str): subscription you wish to use
        location (str): location you wish the strage to be created in
        resource_group_name (str): the name of the resource group you want the storage to be created under
    
    Raises:
        ResourceGroupException: Exception if the resource group could not be created
    
    Returns:
        ResourceGroup: an Azure resource group object
    
    Examples:
    >>> profile = get_cli_profile()
    >>> profile.set_active_subscription("YOUR-SUBSCRIPTION")
    >>> cred, subscription_id, _ = profile.get_login_credentials()
    >>> rg = create_resource_group(cred, subscription_id, "eastus", "testrg2")
    """
    logger = logging.getLogger(__name__)
    resource_group_client = _get_resource_group_client(profile_credentials, subscription_id)
    if resource_group_exists(resource_group_name, resource_group_client=resource_group_client):
        logger.debug(f"Found resource group {resource_group_name}")
        resource_group = resource_group_client.resource_groups.get(resource_group_name)
    else:
        logger.debug(f"Creating resource group {resource_group_name} in {location}")
        resource_group_params = {"location": location}
        resource_group = resource_group_client.resource_groups.create_or_update(
            resource_group_name, resource_group_params
        )

    if "Succeeded" not in resource_group.properties.provisioning_state:
        raise ResourceGroupException(
            f"Resource group not created successfully | State {resource_group.properties.provisioning_state}"
        )

    return resource_group
