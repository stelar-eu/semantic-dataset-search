def compose_dataset_description_for_reranking(dataset_info):
    """
    Compose a complete textual description of a dataset for LLM reranking.
    
    Args:
        dataset_info (dict): Dictionary containing dataset information
        
    Returns:
        str: Complete textual description for reranking
    """
    title = dataset_info.get("dataset_title", "")
    official_desc = dataset_info.get("dataset_official_description", "")
    description = dataset_info.get("dataset_description", "")
    use_case = dataset_info.get("use_case", "")
    domain = dataset_info.get("domain", "")
    
    parts = []
    
    if title:
        parts.append(f"Title: {title}")
    
    if official_desc:
        parts.append(f"Official Description: {official_desc}")
    
    if description:
        parts.append(f"Dataset Description: {description}")
    
    if use_case:
        parts.append(f"Use Case: {use_case}")
    
    if domain:
        parts.append(f"Domain: {domain}")
    
    return "\n".join(parts)