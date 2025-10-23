# This file contains the utility functions for the semantic dataset search API

def flatten_auth_scope(scopes):
    """
    Flatten the auth_scope list into a dictionary for ChromaDB query.
    If no scopes are provided, default to public datasets. This is mainly
    used when searching for datasets.

    Example:
        flatten_auth_scope(["public", "org-a"])
        returns {"auth_scope_public": {"$eq": True}, "auth_scope_org-a": {"$eq": True}}

        flatten_auth_scope([])
        returns {"auth_scope_public": {"$eq": True}}
    """

    if not scopes:
        # Default filtering is show only public datasets
        where_scope = {"auth_scope_public": {"$eq": True}}
    elif len(scopes) == 1:
        # Use simple equality
        where_scope = {f"auth_scope_{scopes[0]}": {"$eq": True}}
    else:
        # Use $or for multiple scopes
        where_scope = {"$or": [{f"auth_scope_{s}": {"$eq": True}} for s in scopes]}
    return where_scope
