import logging
import math
from typing import Dict, Any


class DistanceNormalizer:
    """"
    Utility class for normalizing and calculating distances between vectors or points.

    This class provides methods to normalize distance values using an inverted exponential
    kernel transformation and to extract distance values from search result structures.

    The normalization process scales raw distance values to a range [0,1] using the formula:
        normalized = 1 - exp(-α * d_norm)
    where d_norm is the linearly scaled distance and α is a sharpness parameter (default: 5.0).

    Attributes:
        logger: A logging.Logger instance for debug output.

    Examples:
        >>> normalizer = DistanceNormalizer()
        >>> stats = {'min': 0.0, 'max': 1.0}
        >>> normalizer.normalize_distance(0.5, stats)
        0.9179

        >>> result = {'distances': [0.1, 0.2, 0.3]}
        >>> item = {'metadata': {'model_id': 'model1'}}
        >>> normalizer.extract_search_distance(result, 1, item)
        0.2
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def normalize_distance(self, distance: float, stats: Dict[str, float]) -> float:
        """
        Normalize a distance value using an inverted exponential kernel.

        After linearly scaling into [0,1], we do:
            normalized = 1 - exp(-alpha * d_norm)
        so:
          - d == min → d_norm=0 → normalized=0
          - d == max → d_norm=1 → normalized=1-exp(-alpha)≈1
          - small differences around zero get pulled even closer to zero
        """
        import math

        min_val = stats.get("min", 0.0)
        max_val = stats.get("max", 2.0)

        # avoid division by zero
        if max_val == min_val:
            return 0.0 if distance == min_val else 1.0

        # linear [0,1]
        d0 = (distance - min_val) / (max_val - min_val)
        d0 = max(0.0, min(1.0, d0))

        # invertible exponential kernel
        alpha = 5.0  # higher alpha → sharper falloff
        normalized = 1.0 - math.exp(-alpha * d0)

        self.logger.debug(
            f"Exp-kernel normalize: raw={distance:.4f}, d0={d0:.4f}, "
            f"alpha={alpha}, result={normalized:.4f} (range {min_val}-{max_val})"
        )

        return normalized

    def extract_search_distance(self, result: Dict[str, Any], idx: int, item: Dict[str, Any],
                                table_name: str = 'unknown') -> float:
        """Extract distance from search results."""
        distance = None
        model_id = item.get('metadata', {}).get('model_id', 'unknown')

        if 'distances' in result and isinstance(result['distances'], list):
            if len(result['distances']) > idx:
                if isinstance(result['distances'][idx], list) and len(result['distances'][idx]) > 0:
                    distance = result['distances'][idx][0]  # ChromaDB sometimes returns nested lists
                else:
                    distance = result['distances'][idx]
        else:
            # Or try to get it directly from the item
            distance = item.get('distance')

        # Use a default if all else fails
        if distance is None:
            distance = 2.0

        # Log the distance for debugging
        self.logger.debug(f"Distance for model {model_id} in {table_name}: {distance}")

        return distance