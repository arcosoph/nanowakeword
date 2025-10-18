# utils/helpers.py

# Central configuration for documentation links
DOCS_BASE_URL = "https://github.com/arcosoph/nanowakeword/blob/main/TECHNICAL_GUIDE.md"
ERROR_ANCHORS = {
    "INVALID_MODEL_TYPE": "#model-architecture-err-101",
    "MISSING_BATCH_COMP": "#batch-composition-err-324",
    # Add all future error codes and anchors here
}

def get_doc_link(error_key: str) -> str:
    """
    Generates a deep link to the technical guide for a given error key.
    
    Args:
        error_key (str): The logical name of the error (e.g., "MISSING_BATCH_COMP").
        
    Returns:
        str: The full URL with the specific anchor.
    """
    anchor = ERROR_ANCHORS.get(error_key, "") # Use .get() for safety
    if not anchor:
        print(f"[Warning] Doc link for error '{error_key}' not found. Returning base URL.")
        return DOCS_BASE_URL
    return DOCS_BASE_URL + anchor