import logging
import time
from typing import Dict, Any, List


class NotebookRequestHandler:
    """
    Manager for handling notebook generation requests.
    """

    def __init__(self, chroma_manager, access_control_manager=None, analytics=None):
        """
        Initialize the NotebookManager with required dependencies.

        Args:
            chroma_manager: Manager for Chroma vector database interactions
            access_control_manager: Optional manager for access control
            analytics: Optional analytics collector
        """
        self.chroma_manager = chroma_manager
        self.access_control_manager = access_control_manager
        self.analytics = analytics
        self.logger = logging.getLogger(__name__)

    async def handle_notebook_request(self, query: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a notebook generation request.

        Args:
            query: The processed query text
            parameters: Dictionary of extracted parameters

        Returns:
            Dictionary containing notebook generation results
        """
        self.logger.debug(f"Handling notebook request: {parameters}")
        start_time = time.time()

        try:
            user_id = parameters.get("user_id")
            model_ids = parameters.get("model_ids", [])
            if not model_ids:
                raise ValueError("Notebook generation requires at least one model ID")

            # Filter down to only the models the user can access
            accessible_models = await self._filter_accessible_models(model_ids, user_id)
            if not accessible_models:
                raise ValueError("User does not have access to any of the requested models")

            # Use only permitted models
            model_ids = accessible_models

            analysis_types = parameters.get("analysis_types", ["basic"])
            dataset = parameters.get("dataset", None)
            resources = parameters.get("resources", "standard")

            notebook_request = {
                "model_ids": model_ids,
                "analysis_types": analysis_types,
                "dataset": dataset,
                "resources": resources,
                "user_id": user_id,
            }

            notebook_result = {
                "notebook_id": f"nb_{model_ids[0]}_{int(time.time())}",
                "title": f"Analysis of {', '.join(model_ids)}",
                "status": "pending",
                "estimated_completion_time": int(time.time() + 300),  # 5 minutes from now
            }

            return {
                "success": True,
                "type": "notebook_request",
                "request": notebook_request,
                "result": notebook_result,
                "performance": {
                    "total_time_ms": (time.time() - start_time) * 1000
                },
            }

        except Exception as e:
            self.logger.error(f"Error in notebook request: {e}", exc_info=True)
            raise

    async def _filter_accessible_models(self, model_ids: List[str], user_id: str) -> List[str]:
        """
        Return the subset of model_ids for which the user_id has 'view' access.
        Any search errors are logged and that model_id is skipped.
        """
        accessible = []
        for model_id in model_ids:
            try:
                search_results = await self.chroma_manager.search(
                    collection_name="model_descriptions",
                    query=model_id,
                    include=["metadatas"],
                )
                if not search_results:
                    continue

                for doc in search_results:
                    if doc.get("model_id") != model_id:
                        continue

                    if self.access_control_manager.check_access(doc, user_id, "view"):
                        accessible.append(model_id)
                        break

            except Exception as e:
                self.logger.warning(f"Error checking access for model {model_id}: {e}")
                continue

        return accessible
