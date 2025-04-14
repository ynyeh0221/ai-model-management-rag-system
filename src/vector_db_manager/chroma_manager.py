# src/vector_db_manager/chroma_manager.py

import asyncio
import json
import logging
import os
from typing import Dict, List, Any, Optional

import chromadb
import numpy as np
from chromadb.config import Settings

from src.vector_db_manager.image_embedder import ImageEmbedder
from src.vector_db_manager.text_embedder import TextEmbedder


class ChromaManager:
    """
    Manager for Chroma vector database operations.
    Handles collection management, document operations, and search functionality.
    """

    def __init__(self, text_embedder: TextEmbedder, image_embedder: ImageEmbedder,
                 persist_directory: str = "./chroma_db"):
        """
        Initialize the ChromaManager with database and embedding configuration.
        
        Args:
            persist_directory: Directory for Chroma database persistence
            embedding_model_name: Name of the text embedding model
            image_embedding_model_name: Name of the image embedding model
        """
        self.persist_directory = persist_directory
        self.text_embedder = text_embedder
        self.image_embedder = image_embedder
        self.logger = logging.getLogger(__name__)
        self.collections = {}
        self._initialize_client()

    def _initialize_client(self):
        """
        Initialize the Chroma client and embedding functions.
        Sets up persistent storage and prepares embedding models.
        """
        try:
            # Ensure persistence directory exists
            os.makedirs(self.persist_directory, exist_ok=True)
            
            # Initialize client with persistence settings
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,  # Explicitly disable telemetry
                    allow_reset=True
                )
            )

            # Initialize text embedding function
            self.text_embedding_function = self.text_embedder
            # Initialize image embedding function
            self.image_embedding_function = self.image_embedder
            
            # Initialize default collections
            self._initialize_default_collections()
            
            self.logger.info(f"ChromaManager initialized with persistence directory: {self.persist_directory}")
            
        except Exception as e:
            self.logger.error(f"Error initializing Chroma client: {e}", exc_info=True)
            raise

    def _initialize_default_collections(self):
        """
        Initialize default collections for different document types.
        Sets up optimal index settings for each collection type.
        """
        try:
            # Define default collections
            default_collections = {
                "model_scripts_metadata": {
                    "description": "Collection for model metadata",
                    "embedding_function": self.text_embedding_function,
                    "metadata_schema": {
                        "model_id": "string",
                        "version": "string",
                        "framework": "string",
                        "architecture_type": "string",
                        "created_month": "string",
                        "created_year": "string",
                        "last_modified_month": "string",
                        "last_modified_year": "string",
                        "total_chunks": "number"
                    }
                },
                "model_scripts_chunks": {
                    "description": "Collection for code chunks",
                    "embedding_function": self.text_embedding_function,
                    "metadata_schema": {
                        "model_id": "string",
                        "chunk_id": "number",
                        "total_chunks": "number",
                        "metadata_doc_id": "string",
                        "offset": "number",
                        "type": "string"
                    }
                },
                "generated_images": {
                    "description": "Collection for generated images",
                    "embedding_function": self.image_embedding_function,
                    "metadata_schema": {
                        "source_model_id": "string",
                        "prompt": "string",
                        "style_tags": "string[]",
                        "clip_score": "float"
                    }
                },
                "relationships": {
                    "description": "Collection for relationship documents",
                    "embedding_function": self.text_embedding_function,
                    "metadata_schema": {
                        "source_id": "string",
                        "target_id": "string",
                        "relation_type": "string"
                    }
                }
            }

            # Create or get collections
            for name, config in default_collections.items():
                collection = self._get_or_create_collection(
                    name=name,
                    embedding_function=config["embedding_function"],
                    metadata={"description": config["description"]}
                )

                # Cache the collection for later use
                self.collections[name] = collection

            self.logger.info(f"Initialized {len(default_collections)} default collections")

        except Exception as e:
            self.logger.error(f"Error initializing default collections: {e}", exc_info=True)
            raise

    def _get_or_create_collection(self, name: str,
                                  embedding_function=None,
                                  metadata: Optional[Dict[str, Any]] = None):
        """
        Get or create a collection with the given name.

        Args:
            name: Name of the collection
            embedding_function: Function for generating embeddings
            metadata: Optional metadata for the collection

        Returns:
            The collection object
        """
        try:
            # Check if collection exists
            try:
                collection = self.client.get_collection(
                    name=name,
                    embedding_function=embedding_function
                )
                self.logger.debug(f"Retrieved existing collection: {name}")

            except (ValueError, chromadb.errors.NotFoundError) as e:
                # Collection doesn't exist, create it
                # The specific error type is chromadb.errors.NotFoundError for newer versions
                collection = self.client.create_collection(
                    name=name,
                    embedding_function=embedding_function,
                    metadata=metadata
                )
                self.logger.info(f"Created new collection: {name}")

            # Cache the collection
            self.collections[name] = collection

            return collection

        except Exception as e:
            self.logger.error(f"Error getting or creating collection {name}: {e}", exc_info=True)
            raise
    
    def get_collection(self, name: str):
        """
        Get a collection by name, using cached version if available.
        
        Args:
            name: Name of the collection
            
        Returns:
            The collection object
        """
        if name in self.collections:
            return self.collections[name]
        
        # Determine appropriate embedding function
        if "image" in name.lower():
            embedding_function = self.image_embedding_function
        else:
            embedding_function = self.text_embedding_function
        
        # Get or create the collection
        return self._get_or_create_collection(name, embedding_function,
            metadata={
                "model_id": "string",
                "version": "string",
                "framework": "string",
                "architecture_type": "string"
            })

    async def add_document(self,
                           document: Dict[str, Any],
                           document_id: Optional[str] = None,
                           collection_name: str = "model_scripts",
                           embed_content: bool = True) -> str:
        """
        Add a document to the specified collection.

        Args:
            document: The document to add.
            document_id: Optional document ID. If not provided, one is generated.
            collection_name: Name of the collection to add to.
            embed_content: Whether to generate an embedding using the document's content.

        Returns:
            str: The ID of the added document.
        """
        try:
            # Convert embed_content to a simple boolean if it's a numpy array or other complex type
            if not isinstance(embed_content, bool):
                if isinstance(embed_content, np.ndarray):
                    # If it's a numpy array, convert using any()
                    should_embed = embed_content.any()
                else:
                    try:
                        should_embed = bool(embed_content)
                    except Exception:
                        should_embed = True
            else:
                should_embed = embed_content

            # Use the provided document_id if given, otherwise generate one.
            if document_id is None:
                doc_id = document.get("id", f"{collection_name}_{hash(str(document))}")
            else:
                doc_id = document_id

            content = document.get("content", "")
            metadata = document.get("metadata", {})

            # Ensure metadata is a dictionary.
            if not isinstance(metadata, dict):
                metadata = {}

            # Flatten nested dictionaries in metadata and handle all complex types
            flat_metadata = {}
            for key, value in metadata.items():
                if key == "model_id" and isinstance(value, str):
                    flat_metadata[key] = value  # don't double stringify
                # Safe flattening: only JSON-stringify complex structures
                elif isinstance(value, dict) or isinstance(value, list):
                    flat_metadata[key] = json.dumps(value)
                elif value is None:
                    flat_metadata[key] = ""
                elif isinstance(value, (str, int, float, bool)):
                    flat_metadata[key] = value
                else:
                    flat_metadata[key] = str(value)

            flat_metadata["offset"] = document["metadata"].get("offset", -999)

            # Select the appropriate collection.
            collection = self.get_collection(collection_name)

            # Determine if content is non-empty
            has_content = False
            if isinstance(content, (str, bytes)):
                has_content = bool(content.strip())
            elif isinstance(content, np.ndarray):
                has_content = content.size > 0
            elif hasattr(content, '__len__'):
                has_content = len(content) > 0
            else:
                try:
                    has_content = bool(content)
                except Exception:
                    has_content = False

            # Initialize embeddings to None
            embeddings = None

            # Generate embeddings if requested (or if we're in the images collection)
            if should_embed and (has_content or collection_name == "generated_images"):
                if collection_name == "generated_images":
                    embeddings = await self._run_in_executor(
                        self.image_embedding_function,
                        ["Image embedding placeholder"]
                    )
                else:
                    embeddings = await self._run_in_executor(
                        self.text_embedding_function,
                        [content]
                    )

            # Build keyword arguments for the add() method based on collection type.
            if collection_name == "generated_images":
                # For generated images, expect 'image_path' in metadata.
                image_path = metadata.get("image_path")
                if not image_path:
                    raise ValueError("For 'generated_images' collection, an 'image_path' must be provided in metadata.")
                # Load the image as a NumPy array.
                try:
                    from PIL import Image
                    with Image.open(image_path) as img:
                        # Convert the image to a numpy array.
                        image_np = np.array(img)
                except Exception as e:
                    raise ValueError(f"Error loading image from '{image_path}': {e}")
                add_kwargs = {
                    "ids": [doc_id],
                    "images": [image_np],
                    "embeddings": embeddings,
                    "metadatas": [flat_metadata] if flat_metadata else None
                }
            else:
                add_kwargs = {
                    "ids": [doc_id],
                    "documents": [content] if content else None,
                    "embeddings": embeddings,
                    "metadatas": [flat_metadata] if flat_metadata else None
                }

            # Add the document using the appropriate field.
            await self._run_in_executor(
                collection.add,
                **add_kwargs
            )

            self.logger.debug(f"[DEBUG] Saving document {doc_id} with metadata: {flat_metadata}")

            # Immediately fetch it back
            result = await self.get(
                collection_name=collection_name,
                ids=[doc_id],
                include=["metadatas"]
            )
            print(f"[DEBUG] Immediately fetch result back: {result}")

            return doc_id

        except Exception as e:
            self.logger.error(f"Error adding document to {collection_name}: {e}", exc_info=True)
            raise

    async def add_documents(self, documents: List[Dict[str, Any]],
                          collection_name: str = "model_scripts") -> List[str]:
        """
        Add multiple documents to the specified collection in batch.
        
        Args:
            documents: List of documents to add
            collection_name: Name of the collection to add to
            
        Returns:
            List[str]: IDs of the added documents
        """
        try:
            if not documents:
                return []
            
            # Select the appropriate collection
            collection = self.get_collection(collection_name)
            
            # Extract document components
            ids = []
            contents = []
            metadatas = []
            embeddings_list = []
            
            for document in documents:
                doc_id = document.get("id", f"{collection_name}_{hash(str(document))}")
                content = document.get("content", "")
                metadata = document.get("metadata", {})
                
                # Validate metadata
                if not isinstance(metadata, dict):
                    metadata = {}

                # Flatten metadata like in add_document
                flat_metadata = {}
                for key, value in metadata.items():
                    if isinstance(value, dict):
                        flat_metadata[key] = json.dumps(value)
                    elif isinstance(value, list):
                        flat_metadata[key] = json.dumps(value)
                    elif value is None:
                        flat_metadata[key] = ""
                    elif isinstance(value, (str, int, float, bool)):
                        flat_metadata[key] = value
                    else:
                        flat_metadata[key] = str(value)
                
                ids.append(doc_id)
                contents.append(content)
                metadatas.append(flat_metadata)

                self.logger.debug(f"[DEBUG] Saving document {doc_id} with metadata: {flat_metadata}")

            # Generate embeddings in batch
            if collection_name == "generated_images":
                # Placeholder for image embeddings
                batch_embeddings = await self._run_in_executor(
                    self.image_embedding_function,
                    ["Image embedding placeholder"] * len(documents)
                )
            else:
                batch_embeddings = await self._run_in_executor(
                    self.text_embedding_function,
                    contents
                )
            
            # Add the documents to the collection
            await self._run_in_executor(
                collection.add,
                ids=ids,
                documents=contents,
                embeddings=batch_embeddings,
                metadatas=metadatas
            )
            
            self.logger.debug(f"Added {len(documents)} documents to collection {collection_name}")
            
            return ids
            
        except Exception as e:
            self.logger.error(f"Error adding documents to {collection_name}: {e}", exc_info=True)
            raise

    async def search(self, query: str,
                     collection_name: str = "model_scripts",
                     where: Optional[Dict[str, Any]] = None,
                     limit: int = 100,
                     offset: int = 0,
                     include: Optional[List[str]] = None,
                     user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Search for documents in the specified collection.

        Args:
            query: Query text or dictionary with query parameters
            collection_name: Name of the collection to search
            where: Filter criteria
            limit: Maximum number of results
            offset: Number of results to skip (only used for get() method, not query())
            include: What to include in results (metadata, document, embeddings)
            user_id: Optional user ID for access control

        Returns:
            Dict containing search results
        """
        try:
            # Select the appropriate collection
            collection = self.get_collection(collection_name)

            # If include is not specified, include both metadata and documents
            if include is None:
                include = ["metadatas", "documents", "distances"]  # Correct plural forms

            # Apply access control if user_id is provided
            if user_id is not None and where is not None:
                where = self._apply_access_control(where, user_id)

            # Handle different query types
            if isinstance(query, str):
                # Text query - generate embedding
                if collection_name == "generated_images":
                    # For text-to-image search
                    query_embedding = await self._run_in_executor(
                        self.image_embedding_function,
                        [query]
                    )
                else:
                    query_embedding = await self._run_in_executor(
                        self.text_embedding_function,
                        [query]
                    )

                # Query by embedding
                query_args = {
                    "query_embeddings": query_embedding,
                    "n_results": int(limit) if limit is not None else 100,  # Force conversion to int with fallback
                    "include": include
                }

                # Only add where filter if it's not empty
                if where and len(where) > 0:
                    query_args["where"] = where

                results = await self._run_in_executor(
                    collection.query,
                    **query_args
                )

            else:
                # Metadata-only query with no embedding
                get_args = {
                    "limit": limit,
                    "offset": offset,
                    "include": include
                }

                # Only add where filter if it's not empty
                if where and len(where) > 0:
                    get_args["where"] = where

                results = await self._run_in_executor(
                    collection.get,
                    **get_args
                )

            # Process results into a more user-friendly format
            processed_results = self._process_search_results(results, include)

            self.logger.debug(
                f"Search in {collection_name} returned {len(processed_results.get('results', []))} results")

            return processed_results

        except Exception as e:
            self.logger.error(f"Error searching in {collection_name}: {e}", exc_info=True)
            raise

    async def get(self, collection_name: str = "model_scripts",
                  ids: Optional[List[str]] = None,
                  where: Optional[Dict[str, Any]] = None,
                  limit: Optional[int] = None,
                  offset: Optional[int] = None,
                  include: Optional[List[str]] = None,
                  user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get documents from the specified collection.

        Args:
            collection_name: Name of the collection
            ids: List of document IDs to retrieve
            where: Filter criteria
            limit: Maximum number of results
            offset: Number of results to skip
            include: What to include in results
            user_id: Optional user ID for access control

        Returns:
            Dict containing the retrieved documents
        """
        try:
            # Select the appropriate collection
            collection = self.get_collection(collection_name)

            # If include is not specified, include both metadata and documents
            if include is None:
                include = ["metadatas", "documents"]  # Correct plural forms

            # Apply access control if user_id is provided
            if user_id is not None and where is not None:
                where = self._apply_access_control(where, user_id)

            # Prepare arguments for get method
            get_args = {}

            if ids is not None:
                get_args["ids"] = ids

            if limit is not None:
                get_args["limit"] = limit

            if offset is not None:
                get_args["offset"] = offset

            if include is not None:
                get_args["include"] = include

            # Only add where filter if it's not empty
            if where and len(where) > 0:
                get_args["where"] = where

            print(f"Get args: {get_args}")

            # Get documents
            results = await self._run_in_executor(
                collection.get,
                **get_args
            )

            # Process results into a more user-friendly format
            processed_results = self._process_search_results(results, include)

            self.logger.debug(
                f"Get from {collection_name} returned {len(processed_results.get('results', []))} documents")

            return processed_results

        except Exception as e:
            self.logger.error(f"Error getting documents from {collection_name}: {e}", exc_info=True)
            raise

    async def get_document(self, doc_id: str,
                         collection_name: str = "model_scripts",
                         include: Optional[List[str]] = None,
                         user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get a single document by ID.
        
        Args:
            doc_id: Document ID to retrieve
            collection_name: Name of the collection
            include: What to include in the result
            user_id: Optional user ID for access control
            
        Returns:
            Dict containing the document or None if not found
        """
        try:
            # Get the document
            results = await self.get(
                collection_name=collection_name,
                ids=[doc_id],
                include=include,
                user_id=user_id
            )
            
            # Check if any results were returned
            if not results.get('results'):
                return None
            
            # Return the first (and only) result
            return results['results'][0]
            
        except Exception as e:
            self.logger.error(f"Error getting document {doc_id} from {collection_name}: {e}", exc_info=True)
            raise
    
    async def update_document(self, doc_id: str, 
                            document: Dict[str, Any],
                            collection_name: str = "model_scripts",
                            embed_content: bool = True,
                            user_id: Optional[str] = None) -> bool:
        """
        Update a document by ID.
        
        Args:
            doc_id: Document ID to update
            document: Updated document data
            collection_name: Name of the collection
            embed_content: Whether to generate new embeddings
            user_id: Optional user ID for access control
            
        Returns:
            bool: Success status
        """
        try:
            # Check if document exists and user has access
            existing_doc = await self.get_document(
                doc_id=doc_id,
                collection_name=collection_name,
                user_id=user_id
            )
            
            if not existing_doc:
                self.logger.warning(f"Document {doc_id} not found or user {user_id} does not have access")
                return False
            
            # Extract document components
            content = document.get("content", "")
            metadata = document.get("metadata", {})
            
            # Validate metadata
            if not isinstance(metadata, dict):
                metadata = {}
            
            # Select the appropriate collection
            collection = self.get_collection(collection_name)
            
            # Create embedding if required
            embeddings = None
            if embed_content and content:
                if collection_name == "generated_images":
                    # For real implementation, this would use image embedding
                    embeddings = await self._run_in_executor(
                        self.image_embedding_function,
                        ["Image embedding placeholder"]
                    )
                else:
                    embeddings = await self._run_in_executor(
                        self.text_embedding_function,
                        [content]
                    )
            
            # Update the document
            update_args = {
                "ids": [doc_id],
                "metadatas": [metadata] if metadata else None,
            }
            
            if content:
                update_args["documents"] = [content]
                
            if embeddings:
                update_args["embeddings"] = embeddings
            
            # Run the update
            await self._run_in_executor(
                collection.update,
                **update_args
            )
            
            self.logger.debug(f"Updated document {doc_id} in collection {collection_name}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating document {doc_id} in {collection_name}: {e}", exc_info=True)
            raise
    
    async def delete_document(self, doc_id: str, 
                            collection_name: str = "model_scripts",
                            user_id: Optional[str] = None) -> bool:
        """
        Delete a document by ID.
        
        Args:
            doc_id: Document ID to delete
            collection_name: Name of the collection
            user_id: Optional user ID for access control
            
        Returns:
            bool: Success status
        """
        try:
            # Check if document exists and user has access
            if user_id:
                existing_doc = await self.get_document(
                    doc_id=doc_id,
                    collection_name=collection_name,
                    user_id=user_id
                )
                
                if not existing_doc:
                    self.logger.warning(f"Document {doc_id} not found or user {user_id} does not have access")
                    return False
            
            # Select the appropriate collection
            collection = self.get_collection(collection_name)
            
            # Delete the document
            await self._run_in_executor(
                collection.delete,
                ids=[doc_id]
            )
            
            self.logger.debug(f"Deleted document {doc_id} from collection {collection_name}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting document {doc_id} from {collection_name}: {e}", exc_info=True)
            raise
    
    async def delete_documents(self, where: Dict[str, Any], 
                             collection_name: str = "model_scripts",
                             user_id: Optional[str] = None) -> int:
        """
        Delete documents matching the filter criteria.
        
        Args:
            where: Filter criteria
            collection_name: Name of the collection
            user_id: Optional user ID for access control
            
        Returns:
            int: Number of documents deleted
        """
        try:
            # Apply access control if user_id is provided
            if user_id is not None:
                where = self._apply_access_control(where, user_id)
            
            # Select the appropriate collection
            collection = self.get_collection(collection_name)
            
            # Get documents that match the criteria to count them
            matching_docs = await self.get(
                collection_name=collection_name,
                where=where,
                include=["ids"]
            )
            
            matching_ids = []
            if matching_docs and 'results' in matching_docs:
                matching_ids = [doc.get('id') for doc in matching_docs['results']]
            
            if not matching_ids:
                return 0
            
            # Delete the documents
            await self._run_in_executor(
                collection.delete,
                ids=matching_ids
            )
            
            self.logger.debug(f"Deleted {len(matching_ids)} documents from collection {collection_name}")
            
            return len(matching_ids)
            
        except Exception as e:
            self.logger.error(f"Error batch deleting documents from {collection_name}: {e}", exc_info=True)
            raise
    
    async def _run_in_executor(self, func, *args, **kwargs):
        """
        Run a synchronous function in an executor to make it async-compatible.
        
        Args:
            func: The function to run
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            The result of the function
        """
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: func(*args, **kwargs)
        )

    def _process_search_results(self, results: Dict[str, Any], include: List[str]) -> Dict[str, Any]:
        """
        Process raw Chroma results into a more user-friendly format.
        """
        processed = {"results": []}

        ids = results.get('ids') or []
        documents = results.get('documents') or []
        metadatas = results.get('metadatas') or []
        embeddings = results.get('embeddings') or []
        distances = results.get('distances') or []

        # Flatten structure if necessary
        if isinstance(ids, list) and len(ids) > 0 and isinstance(ids[0], list):
            ids = ids[0]
        if isinstance(documents, list) and len(documents) > 0 and isinstance(documents[0], list):
            documents = documents[0]
        if isinstance(metadatas, list) and len(metadatas) > 0 and isinstance(metadatas[0], list):
            metadatas = metadatas[0]
        if isinstance(embeddings, list) and len(embeddings) > 0 and isinstance(embeddings[0], list):
            embeddings = embeddings[0]
        if isinstance(distances, list) and len(distances) > 0 and isinstance(distances[0], list):
            distances = distances[0]

        result_count = len(ids)
        for i in range(result_count):
            item = {"id": ids[i]}

            if "documents" in include and i < len(documents):
                item["document"] = documents[i]
            if "metadatas" in include or "metadata" in include:
                item["metadata"] = metadatas[i] if i < len(metadatas) and metadatas[i] else {}
            if "embeddings" in include and i < len(embeddings):
                item["embedding"] = embeddings[i]
            if "distances" in include and i < len(distances):
                # Convert distance to score appropriate for cosine distance
                if distances[i] is not None:
                    # For cosine distance (ranges from 0 to 2)
                    # Map to a score from 1.0 (best) to 0.0 (worst)
                    score = max(0.0, 1.0 - (distances[i] / 2))
                else:
                    # Default score if no distance is available
                    score = 0.0
                item["score"] = score

            processed["results"].append(item)

        return processed

    def _apply_access_control(self, where: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """
        Apply access control filters based on user ID.
        
        Args:
            where: Original filter criteria
            user_id: User ID for access control
            
        Returns:
            Dict with updated filter criteria
        """
        # Create a copy to avoid modifying the original
        updated_where = where.copy() if where else {}
        
        # Add access control filter
        # This assumes documents have an access_control.view_permissions field
        # that contains a list of users or groups who can view the document
        access_filter = {
            "$or": [
                {"access_control.owner": {"$eq": user_id}},
                {"access_control.view_permissions": {"$contains": user_id}},
                {"access_control.view_permissions": {"$contains": "public"}}
            ]
        }
        
        # Combine with existing filters
        if not updated_where:
            return access_filter
        
        return {
            "$and": [
                updated_where,
                access_filter
            ]
        }

    async def count_documents(self, collection_name: str = "model_scripts",
                              where: Optional[Dict[str, Any]] = None,
                              user_id: Optional[str] = None) -> int:
        """
        Count documents in a collection.

        Args:
            collection_name: Name of the collection
            where: Filter criteria
            user_id: Optional user ID for access control

        Returns:
            int: Document count
        """
        try:
            # Apply access control if user_id is provided
            if user_id is not None and where is not None:
                where = self._apply_access_control(where, user_id)

            # Get results with just IDs to count them
            results = await self.get(
                collection_name=collection_name,
                where=where,
                include=["ids"]  # This is correct as "ids" not "id"
            )

            count = len(results.get('results', []))

            self.logger.debug(f"Counted {count} documents in collection {collection_name}")

            return count

        except Exception as e:
            self.logger.error(f"Error counting documents in {collection_name}: {e}", exc_info=True)
            raise
    
    async def get_collection_stats(self, collection_name: str = "model_scripts") -> Dict[str, Any]:
        """
        Get statistics for a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Dict with collection statistics
        """
        try:
            # Get the collection
            collection = self.get_collection(collection_name)
            
            # Get collection count
            count = await self._run_in_executor(collection.count)
            
            # Get collection metadata
            # Note: Chroma doesn't provide direct API for this, so we're simulating
            collection_info = {
                "name": collection_name,
                "count": count,
                "description": "Collection for " + collection_name.replace("_", " "),
                "embedding_function": (
                    self.embedding_model_name if "image" not in collection_name
                    else self.image_embedding_model_name
                )
            }
            
            return collection_info
            
        except Exception as e:
            self.logger.error(f"Error getting stats for collection {collection_name}: {e}", exc_info=True)
            raise
