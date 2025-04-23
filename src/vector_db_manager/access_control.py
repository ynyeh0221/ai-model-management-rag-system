import asyncio
import json

"""
# AccessControlManager Documentation

## Overview

The `AccessControlManager` is a Python component that provides fine-grained access control for document-based systems. It allows you to manage who can view or edit documents by attaching permission metadata to documents and enforcing these permissions when accessing documents.

## Key Features

- Permission-based access control (view, edit)
- Support for user-level and group-level permissions
- Public access capabilities
- Query filtering based on permissions
- Support for both synchronous and asynchronous database operations

## Core Concepts

### Permission Hierarchy

The system uses a hierarchical permission model:
- `view`: Basic access to read a document
- `edit`: Advanced access that includes viewing and modifying a document

### Access Control Storage

Access permissions are stored in document metadata as a JSON structure:
```json
{
  "view": ["user1", "group1", "public"],
  "edit": ["user2", "group2"]
}
```

## Usage Examples

### Initializing the Manager

```python
from your_db_client import DatabaseClient
from access_control import AccessControlManager

# Initialize with a database client
db_client = DatabaseClient(connection_string="your_connection_string")
acm = AccessControlManager(db_client=db_client)
```

### Checking Access

```python
# Check if a user can view a document
has_access = acm.check_access(document, user_id="user123", permission_type="view")

# Check if a user can edit a document
can_edit = acm.check_access(document, user_id="user123", permission_type="edit")
```

### Managing Permissions

```python
# Grant view access to a user
acm.grant_access(document_id="doc123", user_id="user456", permission_type="view")

# Revoke edit access from a user
acm.revoke_access(document_id="doc123", user_id="user789", permission_type="edit")

# Make a document publicly viewable
acm.set_public_access(document_id="doc123", permission_type="view", public_access=True)

# Make a document private
acm.set_public_access(document_id="doc123", permission_type="view", public_access=False)
```

### Filtering Queries

```python
# Create a filter for database queries to only return documents the user can access
filter_dict = acm.create_access_filter(user_id="user123")

# Use the filter in a database query
accessible_documents = db_client.query(filter=filter_dict)
```

### Getting Accessible Content

```python
# Get all models accessible to a user
accessible_models = acm.get_accessible_models(user_id="user123")

# Get all images accessible to a user
accessible_images = acm.get_accessible_images(user_id="user123")
```

## Working with Permissions

### Access Control Logic

1. A user has access if they are directly listed in the permission
2. A user has access if any of their groups are listed in the permission
3. Everyone has access if "public" is listed in the permission
4. The "edit" permission automatically includes "view" permission

### Handling Group Memberships

The component checks user group memberships through the `_get_user_groups` method. In a real implementation, this would query a user/group database.

## Handling Asynchronous Operations

The component automatically detects and handles both synchronous and asynchronous database operations:

```python
# Using with an asynchronous client
async def main():
    async_client = AsyncDatabaseClient()
    acm = AccessControlManager(db_client=async_client)
    
    # The component will handle the async nature internally
    await some_function_that_uses_acm(acm)
```

## Error Handling

The component includes comprehensive error handling for database operations, JSON parsing errors, and other potential issues.

## Best Practices

1. Always initialize with a database client for full functionality
2. Use the appropriate permission type for the access level needed
3. Consider using group permissions for easier management of multiple users
4. Use public access cautiously, typically only for view permissions
5. Utilize the access filter for efficient querying of accessible documents

This documentation should help you understand and implement the `AccessControlManager` in your document-based system.
"""
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
            if hasattr(self.db_client, 'get_user'):
                user_record = self.db_client.get_user(user_id)
                if user_record and "groups" in user_record:
                    return user_record["groups"]
        except Exception as e:
            print(f"Error retrieving user groups: {e}")

        return []

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
        Get a list of models that the user has access to by first getting all model IDs from
        the model_date table and then fetching metadata for each accessible model.

        Args:
            user_id: The ID of the user

        Returns:
            list: List of models the user can access with their metadata
        """
        if not self.db_client:
            return []

        try:
            # Step 1: Get all model IDs from model_date table
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
                            collection_name="model_date",
                            include=["metadatas"]
                        )
                        all_models = loop.run_until_complete(coro)
                    except Exception as e:
                        print(f"Error running async db call: {e}")
                        return []
                else:
                    all_models = self.db_client.get(
                        collection_name="model_date",
                        include=["metadatas"]
                    )

                # Define metadata tables to fetch (excluding model_descriptions)
                metadata_tables = [
                    "model_architectures",
                    "model_frameworks",
                    "model_datasets",
                    "model_training_configs",
                    "model_file",
                    "model_git",
                    "model_images_folder"
                ]

                # Process results to collect model IDs
                all_model_ids = set()
                if isinstance(all_models, dict) and 'results' in all_models:
                    for model in all_models['results']:
                        metadata = model.get('metadata', {})
                        model_id = metadata.get('model_id')

                        # Skip if no model_id or not accessible to user
                        if not model_id:
                            continue

                        # Check access control
                        if self.check_access({'metadata': metadata}, user_id, "view"):
                            all_model_ids.add(model_id)

                # Step 2: Fetch complete metadata for accessible models
                accessible_models = []
                seen_model_ids = set()

                for model_id in all_model_ids:
                    try:
                        # Skip if already processed
                        if model_id in seen_model_ids:
                            continue

                        # Initialize consolidated metadata
                        consolidated_metadata = {'model_id': model_id}

                        # Fetch metadata from each table for this model
                        for table_name in metadata_tables:
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
                                            collection_name=table_name,
                                            where={"model_id": {"$eq": model_id}},
                                            include=["metadatas"]
                                        )
                                        table_results = loop.run_until_complete(coro)
                                    except Exception as e:
                                        print(
                                            f"Error running async get for model {model_id} in table {table_name}: {e}")
                                        continue
                                else:
                                    # Synchronous call
                                    table_results = self.db_client.get(
                                        collection_name=table_name,
                                        where={"model_id": {"$eq": model_id}},
                                        include=["metadatas"]
                                    )

                                # Extract and add metadata
                                if table_results and 'results' in table_results and table_results['results']:
                                    metadata = table_results['results'][0].get('metadata', {})

                                    # Add metadata to consolidated record (exclude description field if present)
                                    if 'description' in metadata:
                                        # Create a copy without the description
                                        metadata_without_desc = {k: v for k, v in metadata.items() if
                                                                 k != 'description'}
                                        consolidated_metadata.update(metadata_without_desc)
                                    else:
                                        consolidated_metadata.update(metadata)
                            except Exception as table_e:
                                print(f"Error processing table {table_name} for model {model_id}: {table_e}")

                        # Process file info if available
                        if 'file' in consolidated_metadata and 'images_folder' in consolidated_metadata:
                            try:
                                file_info = json.loads(consolidated_metadata.get('file', '{}'))
                                images_folder_info = json.loads(consolidated_metadata.get('images_folder', '{}'))
                                model_info = {
                                    'model_id': model_id,
                                    'creation_date': file_info.get('creation_date'),
                                    'last_modified_date': file_info.get('last_modified_date'),
                                    'total_chunks': consolidated_metadata.get('total_chunks'),
                                    'absolute_path': file_info.get('absolute_path'),
                                    'images_folder': images_folder_info.get('name'),
                                }

                                # Add other metadata
                                if 'framework' in consolidated_metadata:
                                    model_info['framework'] = consolidated_metadata['framework']
                                if 'version' in consolidated_metadata:
                                    model_info['version'] = consolidated_metadata['version']

                                accessible_models.append(model_info)
                                seen_model_ids.add(model_id)
                            except Exception as inner_e:
                                print(f"Error processing model file info for {model_id}: {inner_e}")
                        else:
                            # If no file info, create a basic model info entry
                            model_info = {
                                'model_id': model_id
                            }
                            # Add all consolidated metadata except large fields
                            for key, value in consolidated_metadata.items():
                                if key not in ['file', 'description', 'access_control']:
                                    model_info[key] = value

                            accessible_models.append(model_info)
                            seen_model_ids.add(model_id)

                    except Exception as e:
                        print(f"Error processing model {model_id}: {e}")

                return accessible_models

            except Exception as e:
                print(f"Error retrieving models from model_date table: {e}")
                return []

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
