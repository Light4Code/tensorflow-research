def get_entry(root_node: dict, entry_name: str, default_value):
    """Gets the entry from the root node or creates one with 
    the default value if none is existing in the root node
    
    Arguments:
        root_node {dict} -- Root node
        entry_name {str} -- Entry name
        default_value {[type]} -- Default value
    
    Returns:
        [type] -- Entry
    """
    node = default_value
    try:
        node = root_node[entry_name]
    except:
        pass
    return node
