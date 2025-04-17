import asyncio
import json


class AccessControlManager:
    def __init__(self, db_client=None):
        """
        Initialize the AccessControlManager with a simplified permission format.

        Args:
            db_client: A client connection to the database that stores access control information.
        """
        self.db_client = db_client
        # Define permission hierarchy - higher levels include lower levels
        self.permission_hierarchy = {
            "edit": ["view", "edit"],  # edit permission includes view permission
            "view": ["view"]  # view permission only includes itself
        }

    def check_access(self, document, user_id, permission_type="view"):
        """
        Check if a user has access to a document.

        Args:
            document: Document object or dict containing access_control metadata
            user_id: The ID of the user requesting access
            permission_type: Type of permission to check ("view", "edit")

        Returns:
            bool: True if user has the required permission, False otherwise
        """
        # Get access control information
        access_control = self._get_access_control(document)
        if not access_control:
            return False

        # Check all permission types that would include the requested permission
        for higher_perm, included_perms in self.permission_hierarchy.items():
            if permission_type in included_perms and higher_perm in access_control:
                permission_list = access_control[higher_perm]

                # Check if user is in the permission list
                if user_id in permission_list:
                    return True

                # Check if any of the user's groups are in the permission list
                user_groups = self._get_user_groups(user_id)
                if any(group in permission_list for group in user_groups):
                    return True

                # Check if "public" is in the permission list, which grants access to everyone
                if "public" in permission_list:
                    return True

        return False

    def grant_access(self, document_id, user_id, permission_type="view"):
        """
        Grant access to a document for a user.

        Args:
            document_id: The ID of the document
            user_id: The ID of the user to grant access to
            permission_type: Type of permission to grant ("view", "edit")

        Returns:
            bool: True if access was granted successfully, False otherwise
        """
        if not self.db_client:
            raise ValueError("Database client is required to modify access control")

        # Retrieve the document
        document = self._get_document(document_id)
        if not document:
            return False

        # Get current access control or initialize
        metadata = document.get("metadata", {})
        access_control_str = metadata.get("access_control", '{"view": [], "edit": []}')

        try:
            access_control = json.loads(access_control_str)
        except json.JSONDecodeError:
            access_control = {"view": [], "edit": []}

        # Ensure the permission type exists
        if permission_type not in access_control:
            access_control[permission_type] = []

        # Add user to the permission list if not already there
        if user_id not in access_control[permission_type]:
            access_control[permission_type].append(user_id)

        # Update the document's metadata
        metadata["access_control"] = json.dumps(access_control)
        return self._update_document_metadata(document_id, metadata)

    def revoke_access(self, document_id, user_id, permission_type="view"):
        """
        Revoke access to a document for a user.

        Args:
            document_id: The ID of the document
            user_id: The ID of the user to revoke access from
            permission_type: Type of permission to revoke ("view", "edit")

        Returns:
            bool: True if access was revoked successfully, False otherwise
        """
        if not self.db_client:
            raise ValueError("Database client is required to modify access control")

        # Retrieve the document
        document = self._get_document(document_id)
        if not document:
            return False

        # Get current access control
        metadata = document.get("metadata", {})
        access_control_str = metadata.get("access_control", '{"view": [], "edit": []}')

        try:
            access_control = json.loads(access_control_str)
        except json.JSONDecodeError:
            return False  # Invalid access_control format

        # Remove user from the permission list if they're in it
        if permission_type in access_control and user_id in access_control[permission_type]:
            access_control[permission_type].remove(user_id)

        # Update the document's metadata
        metadata["access_control"] = json.dumps(access_control)
        return self._update_document_metadata(document_id, metadata)

    def set_public_access(self, document_id, permission_type="view", public_access=True):
        """
        Set or remove public access for a document.

        Args:
            document_id: The ID of the document
            permission_type: Type of permission to set public access for ("view", "edit")
            public_access: True to grant public access, False to revoke it

        Returns:
            bool: True if public access was set/revoked successfully, False otherwise
        """
        if not self.db_client:
            raise ValueError("Database client is required to modify access control")

        # Retrieve the document
        document = self._get_document(document_id)
        if not document:
            return False

        # Get current access control or initialize
        metadata = document.get("metadata", {})
        access_control_str = metadata.get("access_control", '{"view": [], "edit": []}')

        try:
            access_control = json.loads(access_control_str)
        except json.JSONDecodeError:
            access_control = {"view": [], "edit": []}

        # Ensure the permission type exists
        if permission_type not in access_control:
            access_control[permission_type] = []

        # Add or remove "public" from the permission list
        if public_access and "public" not in access_control[permission_type]:
            access_control[permission_type].append("public")
        elif not public_access and "public" in access_control[permission_type]:
            access_control[permission_type].remove("public")

        # Update the document's metadata
        metadata["access_control"] = json.dumps(access_control)
        return self._update_document_metadata(document_id, metadata)

    def create_access_filter(self, user_id):
        """
        Create a filter for access control in queries.
        This generates a filter that can be applied to database queries to only return
        documents the user has permission to view.

        Args:
            user_id: The ID of the user making the query

        Returns:
            dict: A filter dictionary that can be passed to the database query method
        """
        user_groups = self._get_user_groups(user_id)

        # Create a filter that matches documents where:
        # 1. The user is explicitly in any permission list
        # 2. Any of the user's groups are in any permission list
        # 3. The document is public (has "public" in any permission list)

        filter_dict = {
            "$or": [
                {"metadata.access_control": {"$contains": f'"{user_id}"'}},  # User in any permission list
                {"metadata.access_control": {"$contains": '"public"'}}  # Public access
            ]
        }

        # Add group permissions if user belongs to any groups
        if user_groups:
            for group in user_groups:
                filter_dict["$or"].append(
                    {"metadata.access_control": {"$contains": f'"{group}"'}}
                )

        return filter_dict

    def _get_access_control(self, document):
        """
        Get access control information from a document.

        Args:
            document: Document object or dict

        Returns:
            dict: Access control information, or None if it cannot be retrieved
        """
        if not document or "metadata" not in document:
            return None

        metadata = document["metadata"]
        if "access_control" not in metadata:
            return None

        try:
            if isinstance(metadata["access_control"], dict):
                return metadata["access_control"]
            else:
                return json.loads(metadata["access_control"])
        except (json.JSONDecodeError, TypeError):
            return None

    def _get_user_groups(self, user_id):
        """
        Get the groups a user belongs to.

        Args:
            user_id: The ID of the user

        Returns:
            list: List of group IDs the user belongs to
        """
        # In a real implementation, this would query a user/group database
        # For now, we'll return a placeholder implementation

        if not self.db_client:
            return []

        try:
            # Example implementation - adapt based on actual storage mechanism
            user_record = self.db_client.get_user(user_id)
            if user_record and "groups" in user_record:
                return user_record["groups"]
        except Exception as e:
            print(f"Error retrieving user groups: {e}")

        return []

    def _get_document(self, document_id):
        """
        Retrieve a document by ID.

        Args:
            document_id: The ID of the document

        Returns:
            dict: The document if found, None otherwise
        """
        if not self.db_client:
            return None

        try:
            # Check if the get method is asynchronous
            if hasattr(self.db_client, 'get') and asyncio.iscoroutinefunction(self.db_client.get):
                # Handle asynchronous get
                try:
                    # Get or create event loop
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)

                    # Run the async method
                    coro = self.db_client.get(ids=[document_id])
                    return loop.run_until_complete(coro)
                except Exception as e:
                    print(f"Error running async get document: {e}")
                    return None
            else:
                # Synchronous get
                return self.db_client.get(ids=[document_id])
        except Exception as e:
            print(f"Error retrieving document: {e}")
            return None

    def _update_document_metadata(self, document_id, metadata):
        """
        Update a document's metadata.

        Args:
            document_id: The ID of the document
            metadata: The new metadata

        Returns:
            bool: True if update was successful, False otherwise
        """
        if not self.db_client:
            return False

        try:
            # Check if the update method is asynchronous
            if hasattr(self.db_client, 'update') and asyncio.iscoroutinefunction(self.db_client.update):
                # Handle asynchronous update
                try:
                    # Get or create event loop
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)

                    # Run the async method
                    coro = self.db_client.update(
                        ids=[document_id],
                        metadatas=[metadata]
                    )
                    loop.run_until_complete(coro)
                    return True
                except Exception as e:
                    print(f"Error running async update document: {e}")
                    return False
            else:
                # Synchronous update
                self.db_client.update(
                    ids=[document_id],
                    metadatas=[metadata]
                )
                return True
        except Exception as e:
            print(f"Error updating document metadata: {e}")
            return False

    def get_document_permissions(self, document):
        """
        Determine and return the access control metadata for the document.

        Args:
            document (dict): The document for which to determine permissions.

        Returns:
            dict: A dictionary representing access control permissions.
                  For example, {"view": ["public"], "edit": []}
        """
        # Try to get access control from document metadata
        access_control = self._get_access_control(document)
        if access_control:
            return access_control

        # If not found, return default permissions
        return {"view": ["public"], "edit": []}

    def get_accessible_models(self, user_id):
        """
        Get a deduplicated list of models that the user has access to.

        Args:
            user_id: The ID of the user

        Returns:
            list: List of models the user can access
        """
        if not self.db_client:
            return []

        try:
            # Check if db_client.get is an async method
            if hasattr(self.db_client, 'get') and asyncio.iscoroutinefunction(self.db_client.get):
                try:
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)

                    coro = self.db_client.get(
                        collection_name="model_scripts",
                        include=["metadatas"]
                    )
                    all_models = loop.run_until_complete(coro)
                except Exception as e:
                    print(f"Error running async db call: {e}")
                    return []
            else:
                all_models = self.db_client.get(
                    collection_name="model_scripts",
                    include=["metadatas"]
                )

            accessible_models = []
            seen_model_ids = set()

            if isinstance(all_models, dict) and 'results' in all_models:
                for model in all_models['results']:
                    try:
                        metadata = model.get("metadata", {})
                        model_id = metadata.get("model_id")

                        # Skip if already seen or missing
                        if not model_id or model_id in seen_model_ids:
                            continue

                        file_info = json.loads(metadata.get("file", "{}"))

                        if self.check_access(model, user_id, "view"):
                            model_info = {
                                "model_id": model_id,
                                "creation_date": file_info.get("creation_date"),
                                "last_modified_date": file_info.get("last_modified_date"),
                                "total_chunks": metadata.get("total_chunks"),
                                "absolute_path": file_info.get("absolute_path")
                            }

                            # Add other metadata
                            if "framework" in metadata:
                                model_info["framework"] = metadata["framework"]
                            if "version" in metadata:
                                model_info["version"] = metadata["version"]

                            accessible_models.append(model_info)
                            seen_model_ids.add(model_id)

                    except Exception as inner_e:
                        print(f"Error processing model entry: {inner_e}")
                        continue

            return accessible_models

        except Exception as e:
            print(f"Error retrieving accessible models: {e}")
            return []

    def get_accessible_images(self, user_id):
        """
        Get a list of images that the user has access to.

        Args:
            user_id: The ID of the user

        Returns:
            list: List of images the user can access
        """
        if not self.db_client:
            return []

        try:
            # Check if db_client.get is an async method
            if hasattr(self.db_client, 'get') and asyncio.iscoroutinefunction(self.db_client.get):
                # Run async method in a synchronous context
                try:
                    # Get or create an event loop
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)

                    # Create the coroutine
                    coro = self.db_client.get(
                        collection_name="generated_images",
                        include=["metadatas"]
                    )

                    # Run it and get the result
                    all_images = loop.run_until_complete(coro)
                except Exception as e:
                    print(f"Error running async db call: {e}")
                    return []
            else:
                # Synchronous call
                all_images = self.db_client.get(
                    collection_name="generated_images",
                    include=["metadatas"]
                )

            # Process results
            accessible_images = []
            if isinstance(all_images, dict) and 'results' in all_images:
                for image in all_images['results']:
                    if self.check_access(image, user_id, "view"):
                        image_info = {
                            "id": image.get("id", ""),
                            "prompt": image.get("metadata", {}).get("prompt", "No prompt"),
                            "filepath": image.get("metadata", {}).get("image_path", "No path")
                        }
                        # Add other metadata
                        if "metadata" in image and image["metadata"]:
                            metadata = image["metadata"]
                            if "style_tags" in metadata:
                                image_info["style_tags"] = metadata["style_tags"]
                            if "clip_score" in metadata:
                                image_info["clip_score"] = metadata["clip_score"]

                        accessible_images.append(image_info)

            return accessible_images
        except Exception as e:
            print(f"Error retrieving accessible images: {e}")
            return []