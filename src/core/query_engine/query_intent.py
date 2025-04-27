from enum import Enum

# Define intent types as an Enum for type safety
class QueryIntent(Enum):
    RETRIEVAL = "retrieval"  # Basic information retrieval
    NOTEBOOK = "notebook"  # Notebook generation
    IMAGE_SEARCH = "image_search"  # Image search/retrieval
    METADATA = "metadata"  # Metadata-specific queries
    UNKNOWN = "unknown"  # Unknown/ambiguous intent