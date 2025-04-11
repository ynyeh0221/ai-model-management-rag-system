# src/query_engine/result_ranker.py
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable


class ResultRanker:
    """
    Responsible for ranking search results using various strategies.
    """
    
    def __init__(self):
        """Initialize the ResultRanker."""
        self.logger = logging.getLogger(__name__)
        self.ranking_strategies = {
            "similarity": self.similarity_ranking,
            "recency": self.recency_ranking,
            "version": self.version_aware_ranking,
            "hybrid": self.hybrid_ranking
        }
    
    def rank_results(self, 
                    results: List[Dict[str, Any]], 
                    strategy: str = "similarity",
                    custom_ranker: Optional[Callable] = None,
                    **kwargs) -> List[Dict[str, Any]]:
        """
        Rank results using the specified strategy.
        
        Args:
            results: List of result objects to rank
            strategy: Ranking strategy to use ("similarity", "recency", "version", "hybrid")
            custom_ranker: Optional custom ranking function
            **kwargs: Additional parameters for specific ranking strategies
            
        Returns:
            Ranked list of results
        """
        if not results:
            return []
        
        # Copy results to avoid modifying the original list
        results_copy = results.copy()
        
        if custom_ranker:
            # Use custom ranking function if provided
            self.logger.info(f"Using custom ranking function")
            return custom_ranker(results_copy, **kwargs)
        
        if strategy in self.ranking_strategies:
            self.logger.info(f"Ranking results using {strategy} strategy")
            return self.ranking_strategies[strategy](results_copy, **kwargs)
        else:
            self.logger.warning(f"Unknown ranking strategy: {strategy}. Falling back to similarity ranking.")
            return self.similarity_ranking(results_copy)
    
    def similarity_ranking(self, 
                          results: List[Dict[str, Any]],
                          descending: bool = True) -> List[Dict[str, Any]]:
        """
        Rank results based on similarity score.
        
        Args:
            results: List of result objects to rank
            descending: Whether to sort in descending order (highest score first)
            
        Returns:
            Ranked list of results
        """
        # Check if results have 'score' field
        if not all('score' in result for result in results):
            self.logger.warning("Some results missing 'score' field for similarity ranking")
            # Add a default score for results without one
            for result in results:
                if 'score' not in result:
                    result['score'] = 0.0
        
        # Sort by score
        sorted_results = sorted(
            results,
            key=lambda x: x.get('score', 0.0),
            reverse=descending  # Higher score is better in similarity search
        )
        
        return sorted_results
    
    def recency_ranking(self, 
                       results: List[Dict[str, Any]],
                       date_field: str = "metadata.creation_date",
                       descending: bool = True) -> List[Dict[str, Any]]:
        """
        Rank results based on recency.
        
        Args:
            results: List of result objects to rank
            date_field: Field path in the result object that contains the date
            descending: Whether to sort in descending order (newest first)
            
        Returns:
            Ranked list of results
        """
        def _get_date(result, field_path):
            """Extract date from nested dictionary using dot notation field path."""
            parts = field_path.split('.')
            value = result
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    # If path doesn't exist, return None
                    return None
            
            # Convert string date to datetime if needed
            if isinstance(value, str):
                try:
                    return datetime.fromisoformat(value.replace('Z', '+00:00'))
                except (ValueError, TypeError):
                    return None
            elif isinstance(value, datetime):
                return value
            else:
                return None
        
        # Check and extract dates
        dates = []
        for i, result in enumerate(results):
            date = _get_date(result, date_field)
            if date is None:
                self.logger.warning(f"Result {i} missing or invalid date in field '{date_field}'")
                # Use epoch start as default for sorting
                date = datetime(1970, 1, 1)
            dates.append((i, date))
        
        # Sort by date
        sorted_indices = [i for i, date in sorted(dates, key=lambda x: x[1], reverse=descending)]
        
        # Rearrange results
        sorted_results = [results[i] for i in sorted_indices]
        
        return sorted_results
    
    def version_aware_ranking(self, 
                             results: List[Dict[str, Any]],
                             model_id_field: str = "metadata.model_id",
                             version_field: str = "metadata.version") -> List[Dict[str, Any]]:
        """
        Rank results with preference for newer versions of the same model.
        
        This ranking will group results by model family and then prioritize 
        newer versions within each family.
        
        Args:
            results: List of result objects to rank
            model_id_field: Field path for model ID
            version_field: Field path for version
            
        Returns:
            Ranked list of results
        """
        def _get_nested_value(result, field_path):
            """Extract value from nested dictionary using dot notation field path."""
            parts = field_path.split('.')
            value = result
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    # If path doesn't exist, return None
                    return None
            return value
        
        def _parse_version(version_str):
            """Parse version string into comparable components."""
            if not version_str or not isinstance(version_str, str):
                return (0, 0, 0)
            
            # Remove 'v' prefix if present
            if version_str.lower().startswith('v'):
                version_str = version_str[1:]
            
            # Split by dots
            parts = version_str.split('.')
            components = []
            
            # Convert parts to integers, filling with zeros if needed
            for part in parts:
                try:
                    components.append(int(part))
                except ValueError:
                    # If part contains non-numeric characters, stop parsing
                    break
            
            # Ensure we have at least 3 components (major, minor, patch)
            while len(components) < 3:
                components.append(0)
            
            return tuple(components)
        
        # Group results by model family/ID
        model_groups = {}
        for result in results:
            model_id = _get_nested_value(result, model_id_field)
            # Extract base model ID (without version suffix if present)
            base_model_id = model_id.split('-v')[0] if model_id and isinstance(model_id, str) else 'unknown'
            
            if base_model_id not in model_groups:
                model_groups[base_model_id] = []
            
            model_groups[base_model_id].append(result)
        
        # Sort each group by version
        for base_model_id in model_groups:
            model_groups[base_model_id].sort(
                key=lambda x: _parse_version(_get_nested_value(x, version_field)),
                reverse=True  # Higher version first
            )
        
        # Flatten the groups, interleaving results for diversity
        sorted_results = []
        while any(model_groups.values()):
            for base_model_id in list(model_groups.keys()):
                if model_groups[base_model_id]:
                    sorted_results.append(model_groups[base_model_id].pop(0))
        
        return sorted_results
    
    def hybrid_ranking(self, 
                      results: List[Dict[str, Any]], 
                      weights: Optional[Dict[str, float]] = None,
                      **kwargs) -> List[Dict[str, Any]]:
        """
        Rank results using a weighted combination of strategies.
        
        Args:
            results: List of result objects to rank
            weights: Dictionary mapping strategy names to weights
                     e.g., {"similarity": 0.6, "recency": 0.3, "version": 0.1}
            **kwargs: Additional parameters for specific ranking strategies
            
        Returns:
            Ranked list of results
        """
        if not weights:
            weights = {
                "similarity": 0.6,
                "recency": 0.3,
                "version": 0.1
            }
        
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        normalized_weights = {k: v / total_weight for k, v in weights.items()}
        
        # Compute ranking scores for each strategy
        strategy_rankings = {}
        for strategy, weight in normalized_weights.items():
            if weight > 0 and strategy in self.ranking_strategies and strategy != "hybrid":
                # Get ranking for this strategy
                ranked = self.ranking_strategies[strategy](results.copy(), **kwargs)
                
                # Convert to positions (0 = best, len-1 = worst)
                positions = {r["id"]: i for i, r in enumerate(ranked)}
                strategy_rankings[strategy] = positions
        
        # Compute combined score for each result
        result_scores = []
        for i, result in enumerate(results):
            result_id = result.get("id", f"unknown-{i}")
            
            # Calculate normalized position score (0 to 1, where 1 is best)
            # for each strategy
            strategy_scores = {}
            num_results = len(results)
            
            for strategy, positions in strategy_rankings.items():
                if result_id in positions:
                    # Convert position to score (reverse and normalize)
                    position = positions[result_id]
                    strategy_scores[strategy] = 1 - (position / max(1, num_results - 1))
                else:
                    strategy_scores[strategy] = 0
            
            # Compute weighted score
            weighted_score = sum(
                strategy_scores.get(strategy, 0) * weight
                for strategy, weight in normalized_weights.items()
            )
            
            result_scores.append((i, weighted_score))
        
        # Sort by score (highest first)
        sorted_indices = [i for i, score in sorted(result_scores, key=lambda x: x[1], reverse=True)]
        
        # Rearrange results
        sorted_results = [results[i] for i in sorted_indices]
        
        return sorted_results
    
    def diversity_reranking(self, 
                           results: List[Dict[str, Any]], 
                           diversity_fields: List[str] = None,
                           diversity_weight: float = 0.3) -> List[Dict[str, Any]]:
        """
        Re-rank results to improve diversity in the top results.
        
        Args:
            results: List of result objects to rank (already ranked by another strategy)
            diversity_fields: List of fields to consider for diversity
            diversity_weight: Weight given to diversity (0 to 1)
            
        Returns:
            Re-ranked list of results
        """
        if not results or len(results) <= 1:
            return results
        
        if not diversity_fields:
            diversity_fields = ["metadata.model_family", "metadata.framework.name"]
        
        def _get_field_value(result, field_path):
            """Extract value from nested dictionary using dot notation field path."""
            parts = field_path.split('.')
            value = result
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return None
            return value
        
        # Initialize the final ranked list with the top result
        reranked = [results[0]]
        remaining = results[1:]
        
        # Iteratively select the next result with maximum diversity
        while remaining:
            # For each candidate, calculate diversity score
            max_diversity_score = -1
            max_diversity_index = 0
            
            for i, candidate in enumerate(remaining):
                # Calculate how different this candidate is from already selected results
                diversity_score = 0
                
                for field in diversity_fields:
                    candidate_value = _get_field_value(candidate, field)
                    
                    # Count how many already selected results have the same value
                    if candidate_value is not None:
                        same_value_count = sum(
                            1 for r in reranked 
                            if _get_field_value(r, field) == candidate_value
                        )
                        
                        # Normalize by the number of results already selected
                        diversity_score += 1 - (same_value_count / len(reranked))
                
                # Normalize by the number of fields
                diversity_score /= max(1, len(diversity_fields))
                
                # Calculate position score (favoring originally higher-ranked results)
                position_score = 1 - (i / max(1, len(remaining) - 1))
                
                # Combine diversity and position scores
                final_score = (diversity_weight * diversity_score) + ((1 - diversity_weight) * position_score)
                
                if final_score > max_diversity_score:
                    max_diversity_score = final_score
                    max_diversity_index = i
            
            # Add the most diverse result to the final list
            reranked.append(remaining.pop(max_diversity_index))
        
        return reranked

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Sample results
    sample_results = [
        {
            "id": "model1",
            "score": 0.85,
            "metadata": {
                "model_id": "transformer-v1",
                "model_family": "transformer",
                "version": "1.0.0",
                "creation_date": "2023-01-15T10:30:00",
                "framework": {
                    "name": "pytorch"
                }
            }
        },
        {
            "id": "model2",
            "score": 0.92,
            "metadata": {
                "model_id": "transformer-v2",
                "model_family": "transformer",
                "version": "2.0.0",
                "creation_date": "2023-05-20T14:45:00",
                "framework": {
                    "name": "pytorch"
                }
            }
        },
        {
            "id": "model3",
            "score": 0.78,
            "metadata": {
                "model_id": "cnn-v1",
                "model_family": "cnn",
                "version": "1.0.0",
                "creation_date": "2023-03-10T09:15:00",
                "framework": {
                    "name": "tensorflow"
                }
            }
        }
    ]
    
    # Initialize ranker
    ranker = ResultRanker()
    
    # Test different ranking strategies
    print("Similarity ranking:")
    for result in ranker.rank_results(sample_results, strategy="similarity"):
        print(f"  {result['id']} (score: {result['score']})")
    
    print("\nRecency ranking:")
    for result in ranker.rank_results(sample_results, strategy="recency"):
        print(f"  {result['id']} (date: {result['metadata']['creation_date']})")
    
    print("\nVersion-aware ranking:")
    for result in ranker.rank_results(sample_results, strategy="version"):
        print(f"  {result['id']} (version: {result['metadata']['version']})")
    
    print("\nHybrid ranking:")
    for result in ranker.rank_results(sample_results, strategy="hybrid"):
        print(f"  {result['id']}")
    
    print("\nDiversity re-ranking:")
    for result in ranker.diversity_reranking(ranker.rank_results(sample_results, strategy="similarity")):
        print(f"  {result['id']} (family: {result['metadata']['model_family']}, framework: {result['metadata']['framework']['name']})")
