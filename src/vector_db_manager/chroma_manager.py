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
                # Original collections for backward compatibility
                "model_scripts_metadata": {
                    "description": "Collection for model metadata (legacy)",
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
                },

                # New collections for separated metadata
                "model_file": {
                    "description": "Collection for model file metadata",
                    "embedding_function": self.text_embedding_function,
                    "metadata_schema": {
                        "model_id": "string",
                        "file": "string"  # Stored as JSON string
                    }
                },
                "model_date": {
                    "description": "Collection for model date information",
                    "embedding_function": self.text_embedding_function,
                    "metadata_schema": {
                        "model_id": "string",
                        "created_at": "string",
                        "created_month": "string",
                        "created_year": "string",
                        "last_modified_month": "string",
                        "last_modified_year": "string"
                    }
                },
                "model_git": {
                    "description": "Collection for model git information",
                    "embedding_function": self.text_embedding_function,
                    "metadata_schema": {
                        "model_id": "string",
                        "git": "string"  # Stored as JSON string
                    }
                },
                "model_frameworks": {
                    "description": "Collection for model frameworks information",
                    "embedding_function": self.text_embedding_function,
                    "metadata_schema": {
                        "model_id": "string",
                        "framework": "string"  # Stored as JSON string
                    }
                },
                "model_datasets": {
                    "description": "Collection for model datasets information",
                    "embedding_function": self.text_embedding_function,
                    "metadata_schema": {
                        "model_id": "string",
                        "dataset": "string"  # Stored as JSON string
                    }
                },
                "model_training_configs": {
                    "description": "Collection for model training configurations",
                    "embedding_function": self.text_embedding_function,
                    "metadata_schema": {
                        "model_id": "string",
                        "training_config": "string"  # Stored as JSON string
                    }
                },
                "model_architectures": {
                    "description": "Collection for model architecture information",
                    "embedding_function": self.text_embedding_function,
                    "metadata_schema": {
                        "model_id": "string",
                        "architecture": "string"  # Stored as JSON string
                    }
                },
                "model_descriptions": {
                    "description": "Collection for model descriptions",
                    "embedding_function": self.text_embedding_function,
                    "metadata_schema": {
                        "model_id": "string",
                        "description": "string",
                        "total_chunks": "number",
                        "offset": "number"
                    }
                },
                "model_ast_summaries": {
                    "description": "Collection for model AST summary information",
                    "embedding_function": self.text_embedding_function,
                    "metadata_schema": {
                        "model_id": "string",
                        "ast_summary": "string"
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

            self.logger.info(f"Initialized {len(default_collections)} collections")

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

        Note:
            This method uses collection.upsert internally, which means if a document
            with the same ID already exists, it will be replaced with the new document.
            This enables updating existing documents by using the same document_id.
        """
        try:
            # Process document ID
            doc_id = self._get_document_id(document, document_id, collection_name)

            # Process content and metadata
            content = document.get("content", "")
            metadata = document.get("metadata", {})
            flat_metadata = self._flatten_metadata(metadata)

            # Get collection and prepare for upsert
            collection = self.get_collection(collection_name)
            embeddings = await self._generate_embeddings(content, collection_name, embed_content)

            # Add document to collection
            await self._upsert_document(
                collection,
                doc_id,
                content,
                embeddings,
                flat_metadata,
                collection_name,
                document
            )
            # Immediately fetch back for verification
            await self.get(collection_name, [doc_id])

            self.logger.debug(f"[DEBUG] Saved document {doc_id} with metadata: {flat_metadata}")

            return doc_id

        except Exception as e:
            self.logger.error(f"Error adding document to {collection_name}: {e}", exc_info=True)
            raise

    def _get_document_id(self, document: Dict[str, Any], document_id: Optional[str], collection_name: str) -> str:
        """Generate or use provided document ID."""
        if document_id is not None:
            return document_id
        return document.get("id", f"{collection_name}_{hash(str(document))}")

    def _flatten_metadata(self, metadata: Any) -> Dict[str, Any]:
        """Flatten nested dictionaries in metadata and handle complex types."""
        if not isinstance(metadata, dict):
            return {}

        flat_metadata = {}
        for key, value in metadata.items():
            if key == "model_id" and isinstance(value, str):
                flat_metadata[key] = value  # don't double stringify
            # Safe flattening: only JSON-stringify complex structures
            elif isinstance(value, (dict, list)):
                flat_metadata[key] = json.dumps(value)
            elif value is None:
                flat_metadata[key] = ""
            elif isinstance(value, (str, int, float, bool)):
                flat_metadata[key] = value
            else:
                flat_metadata[key] = str(value)

        flat_metadata["offset"] = metadata.get("offset", -999)
        return flat_metadata

    async def _generate_embeddings(self, content, collection_name: str, embed_content) -> Optional[List[Any]]:
        """Generate embeddings based on content and collection type."""
        # Normalize embed_content to boolean
        should_embed = self._normalize_embed_flag(embed_content)

        # Check if content exists
        has_content = self._check_content_exists(content)

        # Generate embeddings if needed
        if should_embed and (has_content or collection_name == "generated_images"):
            if collection_name == "generated_images":
                return await self._run_in_executor(
                    self.image_embedding_function,
                    ["Image embedding placeholder"]
                )
            else:
                return await self._run_in_executor(
                    self.text_embedding_function,
                    [content]
                )
        return None

    def _normalize_embed_flag(self, embed_content) -> bool:
        """Convert embed_content to a simple boolean."""
        if not isinstance(embed_content, bool):
            if isinstance(embed_content, np.ndarray):
                return embed_content.any()
            try:
                return bool(embed_content)
            except Exception:
                return True
        return embed_content

    def _check_content_exists(self, content) -> bool:
        """Determine if content is non-empty."""
        if isinstance(content, (str, bytes)):
            return bool(content.strip())
        elif isinstance(content, np.ndarray):
            return content.size > 0
        elif hasattr(content, '__len__'):
            return len(content) > 0
        else:
            try:
                return bool(content)
            except Exception:
                return False

    async def _upsert_document(self, collection, doc_id: str, content, embeddings,
                               flat_metadata: Dict[str, Any], collection_name: str,
                               document: Dict[str, Any]) -> None:
        """Upsert document into collection with appropriate fields."""
        if collection_name == "generated_images":
            add_kwargs = self._prepare_image_upsert_args(doc_id, flat_metadata, embeddings, document)
        else:
            add_kwargs = {
                "ids": [doc_id],
                "documents": [content] if content else None,
                "embeddings": embeddings,
                "metadatas": [flat_metadata] if flat_metadata else None
            }

        await self._run_in_executor(
            collection.upsert,
            **add_kwargs
        )

    def _prepare_image_upsert_args(self, doc_id: str, flat_metadata: Dict[str, Any],
                                   embeddings, document: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare arguments for upserting images."""
        metadata = document.get("metadata", {})
        image_path = metadata.get("image_path")

        if not image_path:
            raise ValueError("For 'generated_images' collection, an 'image_path' must be provided in metadata.")

        # Load the image as a NumPy array
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                image_np = np.array(img)
        except Exception as e:
            raise ValueError(f"Error loading image from '{image_path}': {e}")

        return {
            "ids": [doc_id],
            "images": [image_np],
            "embeddings": embeddings,
            "metadatas": [flat_metadata] if flat_metadata else None
        }

    async def add_documents(self, documents: List[Dict[str, Any]],
                            collection_name: str = "model_scripts") -> List[str]:
        """
        Add multiple documents to the specified collection in batch.

        Args:
            documents: List of documents to add
            collection_name: Name of the collection to add to

        Returns:
            List[str]: IDs of the added documents

        Note:
            This method uses collection.upsert internally for batch operations, which means
            if documents with the same IDs already exist, they will be replaced with the new
            documents. This enables updating existing documents in bulk.
        """
        try:
            if not documents:
                return []

            # Select the appropriate collection
            collection = self.get_collection(collection_name)

            # Process documents in batch
            processed_docs = self._process_documents_batch(documents, collection_name)

            # Generate embeddings in batch
            batch_embeddings = await self._generate_batch_embeddings(
                processed_docs["contents"],
                collection_name,
                len(documents)
            )

            # Add the documents to the collection
            await self._run_in_executor(
                collection.upsert,
                ids=processed_docs["ids"],
                documents=processed_docs["contents"],
                embeddings=batch_embeddings,
                metadatas=processed_docs["metadatas"]
            )

            self.logger.debug(f"Added {len(documents)} documents to collection {collection_name}")

            return processed_docs["ids"]

        except Exception as e:
            self.logger.error(f"Error adding documents to {collection_name}: {e}", exc_info=True)
            raise

    def _process_documents_batch(self, documents: List[Dict[str, Any]],
                                 collection_name: str) -> Dict[str, List]:
        """Process a batch of documents to prepare for upsert."""
        ids = []
        contents = []
        metadatas = []

        for document in documents:
            # Get document ID
            doc_id = self._get_document_id(document, None, collection_name)

            # Extract content and metadata
            content = document.get("content", "")
            metadata = document.get("metadata", {})

            # Flatten metadata
            flat_metadata = self._flatten_metadata(metadata)

            # Collect document components
            ids.append(doc_id)
            contents.append(content)
            metadatas.append(flat_metadata)

            self.logger.debug(f"[DEBUG] Prepared document {doc_id} with metadata: {flat_metadata}")

        return {
            "ids": ids,
            "contents": contents,
            "metadatas": metadatas
        }

    async def _generate_batch_embeddings(self, contents: List[str],
                                         collection_name: str,
                                         batch_size: int) -> List[Any]:
        """Generate embeddings for a batch of documents."""
        if collection_name == "generated_images":
            # Placeholder for image embeddings
            return await self._run_in_executor(
                self.image_embedding_function,
                ["Image embedding placeholder"] * batch_size
            )
        else:
            return await self._run_in_executor(
                self.text_embedding_function,
                contents
            )

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

            # print(f"Get results: {results}")

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
        distances = results.get('distances') or []  # Make sure to extract distances

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
            distances = distances[0]  # Flatten distances if nested

        result_count = len(ids)
        for i in range(result_count):
            item = {"id": ids[i]}

            if "documents" in include and i < len(documents):
                item["document"] = documents[i]
            if "metadatas" in include or "metadata" in include:
                item["metadata"] = metadatas[i] if i < len(metadatas) and metadatas[i] else {}
            if "embeddings" in include and i < len(embeddings):
                item["embedding"] = embeddings[i]
            if "distances" in include and i < len(distances):  # Add distances to results
                item["distance"] = distances[i]

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
            collection = self.get_collection(collection_name)
            count = await self._run_in_executor(collection.count)

            collection_info = {
                "name": collection_name,
                "count": count,
                "description": "Collection for " + collection_name.replace("_", " "),
                "embedding_function": (
                    "text_embedder" if "image" not in collection_name else "image_embedder"
                )
            }

            return collection_info

        except Exception as e:
            self.logger.error(f"Error getting stats for collection {collection_name}: {e}", exc_info=True)
            raise
