import asyncio
import json

# ────────────────────────────────────────────────────────────────────────────────
# Constants (to avoid duplicated string literals)
# ────────────────────────────────────────────────────────────────────────────────

_DB_CLIENT_REQUIRED_ERROR = "Database client is required to modify access control"
_DEFAULT_ACCESS_CONTROL_JSON = '{"view": [], "edit": []}'
_METADATA_ACCESS_CONTROL = "metadata.access_control"
_CONTAINS_OPERATOR = "$contains"


class AccessControlManager:
    """
    Manages fine‐grained access control for document‐based systems.
    """

    def __init__(self, db_client=None):
        """
        Initialize the AccessControlManager with a DB client.

        Args:
            db_client: A client connection to the database that stores access control info.
        """
        self.db_client = db_client

        # Permission hierarchy: higher levels include lower levels
        self.permission_hierarchy = {
            "edit": ["view", "edit"],  # 'edit' implies 'view'
            "view": ["view"]
        }

    # ────────────────────────────────────────────────────────────────────────────────
    # Public: Check / Grant / Revoke / Public‐access methods
    # ────────────────────────────────────────────────────────────────────────────────

    def check_access(self, document, user_id, permission_type="view"):
        """
        Check if a user has a given permission on a document.

        Args:
            document: Document object or dict (must have 'metadata.access_control').
            user_id: The ID of the user requesting access.
            permission_type: "view" or "edit".

        Returns:
            bool: True if the user has the required permission; False otherwise.
        """
        access_control = self._get_access_control(document)
        if access_control is None:
            return False

        for higher_perm, includes in self.permission_hierarchy.items():
            # Only consider permission branches that include requested permission_type
            if permission_type in includes and higher_perm in access_control:
                perm_list = access_control[higher_perm]

                # Direct user check
                if user_id in perm_list:
                    return True

                # Group membership check
                for group in self._get_user_groups(user_id):
                    if group in perm_list:
                        return True

                # Public access check
                if "public" in perm_list:
                    return True

        return False

    def grant_access(self, document_id, user_id, permission_type="view"):
        """
        Grant a permission (view/edit) to a user on a document.

        Args:
            document_id: The ID of the document.
            user_id: The ID of the user to grant access to.
            permission_type: "view" or "edit".

        Returns:
            bool: True if granted successfully; False otherwise.
        """
        if not self.db_client:
            raise ValueError(_DB_CLIENT_REQUIRED_ERROR)

        document = self._get_document(document_id)
        if document is None:
            return False

        metadata = document.get("metadata", {})
        access_control_str = metadata.get("access_control", _DEFAULT_ACCESS_CONTROL_JSON)

        try:
            access_control = json.loads(access_control_str)
        except json.JSONDecodeError:
            access_control = {"view": [], "edit": []}

        # Ensure the permission_type key exists
        access_control.setdefault(permission_type, [])

        if user_id not in access_control[permission_type]:
            access_control[permission_type].append(user_id)

        metadata["access_control"] = json.dumps(access_control)
        return self._update_document_metadata(document_id, metadata)

    def revoke_access(self, document_id, user_id, permission_type="view"):
        """
        Revoke a permission (view/edit) from a user on a document.

        Args:
            document_id: The ID of the document.
            user_id: The ID of the user to revoke access from.
            permission_type: "view" or "edit".

        Returns:
            bool: True if revoked successfully; False otherwise.
        """
        if not self.db_client:
            raise ValueError(_DB_CLIENT_REQUIRED_ERROR)

        document = self._get_document(document_id)
        if document is None:
            return False

        metadata = document.get("metadata", {})
        access_control_str = metadata.get("access_control", _DEFAULT_ACCESS_CONTROL_JSON)

        try:
            access_control = json.loads(access_control_str)
        except json.JSONDecodeError:
            return False

        if permission_type in access_control and user_id in access_control[permission_type]:
            access_control[permission_type].remove(user_id)

        metadata["access_control"] = json.dumps(access_control)
        return self._update_document_metadata(document_id, metadata)

    def set_public_access(self, document_id, permission_type="view", public_access=True):
        """
        Toggle public access for a permission type on a document.

        Args:
            document_id: The ID of the document.
            permission_type: "view" or "edit".
            public_access: True to grant public access; False to revoke it.

        Returns:
            bool: True if the operation succeeded; False otherwise.
        """
        if not self.db_client:
            raise ValueError(_DB_CLIENT_REQUIRED_ERROR)

        document = self._get_document(document_id)
        if document is None:
            return False

        metadata = document.get("metadata", {})
        access_control_str = metadata.get("access_control", _DEFAULT_ACCESS_CONTROL_JSON)

        try:
            access_control = json.loads(access_control_str)
        except json.JSONDecodeError:
            access_control = {"view": [], "edit": []}

        access_control.setdefault(permission_type, [])

        if public_access:
            if "public" not in access_control[permission_type]:
                access_control[permission_type].append("public")
        else:
            if "public" in access_control[permission_type]:
                access_control[permission_type].remove("public")

        metadata["access_control"] = json.dumps(access_control)
        return self._update_document_metadata(document_id, metadata)

    def get_document_permissions(self, document):
        """
        Return the access control permissions for a document, or default if none set.

        Args:
            document (dict): The document whose permissions to retrieve.

        Returns:
            dict: Permissions dict, e.g. {"view": ["public"], "edit": []}.
        """
        access_control = self._get_access_control(document)
        if access_control:
            return access_control
        return {"view": ["public"], "edit": []}

    # ────────────────────────────────────────────────────────────────────────────────
    # Public: Access‐filter and retrieving accessible resources
    # ────────────────────────────────────────────────────────────────────────────────

    def create_access_filter(self, user_id):
        """
        Create a Mongo‐style filter dict restricting query results to documents the user can view.

        Args:
            user_id: The ID of the user making the query.

        Returns:
            dict: A filter dictionary suitable for passing to the DB query method.
        """
        user_groups = self._get_user_groups(user_id)

        # Base filter conditions: user OR public
        or_conditions = [
            { _METADATA_ACCESS_CONTROL: { _CONTAINS_OPERATOR: f'\"{user_id}\"' }},
            { _METADATA_ACCESS_CONTROL: { _CONTAINS_OPERATOR: '\"public\"' }}
        ]

        # Add group conditions if applicable
        for group in user_groups:
            or_conditions.append({ _METADATA_ACCESS_CONTROL: { _CONTAINS_OPERATOR: f'\"{group}\"' }})

        return { "$or": or_conditions }

    def get_accessible_models(self, user_id):
        """
        Return a list of model metadata dicts that the user has 'view' access to.

        Args:
            user_id: The ID of the user.

        Returns:
            list of dict: Each dict contains consolidated metadata for an accessible model.
        """
        if not self.db_client:
            return []

        # Step 1: Fetch raw model entries from 'model_date' collection
        all_models = self._fetch_all_model_metadata()

        # Step 2: Determine which model_ids the user can view
        accessible_ids = self._filter_accessible_model_ids(all_models, user_id)
        if not accessible_ids:
            return []

        # Step 3: For each accessible model_id, gather consolidated metadata
        result = []
        for model_id in accessible_ids:
            info = self._consolidate_model_info(model_id)
            if info:
                result.append(info)

        return result

    def get_accessible_images(self, user_id):
        """
        Return a list of image entries that the user has 'view' access to.

        Args:
            user_id: The ID of the user.

        Returns:
            list of dict: Each dict contains 'id', 'prompt', 'filepath', and other metadata.
        """
        if not self.db_client:
            return []

        # Step 1: Fetch all 'generated_images' entries
        all_images = self._fetch_all_images()

        # Step 2: Filter them by access_control
        return self._filter_accessible_images(all_images, user_id)

    # ────────────────────────────────────────────────────────────────────────────────
    # Private: Low‐level helpers (each kept simple to reduce complexity)
    # ────────────────────────────────────────────────────────────────────────────────

    def _get_access_control(self, document):
        """
        Extract and parse the access_control JSON from a document.

        Args:
            document: Document object or dict.

        Returns:
            dict or None: Parsed access_control, or None if missing/invalid.
        """
        if not document or "metadata" not in document:
            return None

        metadata = document["metadata"]
        raw = metadata.get("access_control")
        if raw is None:
            return None

        if isinstance(raw, dict):
            return raw

        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None

    def _get_user_groups(self, user_id):
        """
        Retrieve the list of group IDs for a given user.

        Args:
            user_id: The ID of the user.

        Returns:
            list: List of group IDs (empty list if none or on error).
        """
        if not self.db_client:
            return []

        try:
            if hasattr(self.db_client, "get_user"):
                user_record = self.db_client.get_user(user_id)
                if user_record and "groups" in user_record:
                    return user_record["groups"]
        except Exception:
            pass

        return []

    def _get_document(self, document_id):
        """
        Retrieve a document by ID, handling both async and sync DB clients.

        Args:
            document_id: The ID of the document.

        Returns:
            dict or None: The document if found, else None.
        """
        try:
            if hasattr(self.db_client, "get") and asyncio.iscoroutinefunction(self.db_client.get):
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                coro = self.db_client.get(ids=[document_id])
                return loop.run_until_complete(coro)
            else:
                return self.db_client.get(ids=[document_id])
        except Exception:
            return None

    def _update_document_metadata(self, document_id, metadata):
        """
        Update only the metadata of a document, handling async vs. sync.

        Args:
            document_id: The ID of the document.
            metadata: Dict of metadata to set.

        Returns:
            bool: True if successful; False otherwise.
        """
        try:
            if hasattr(self.db_client, "update") and asyncio.iscoroutinefunction(self.db_client.update):
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                coro = self.db_client.update(ids=[document_id], metadatas=[metadata])
                loop.run_until_complete(coro)
                return True
            else:
                self.db_client.update(ids=[document_id], metadatas=[metadata])
                return True
        except Exception:
            return False

    # ────────────────────────────────────────────────────────────────────────────────
    # Private: get_accessible_models helpers
    # ────────────────────────────────────────────────────────────────────────────────

    def _fetch_all_model_metadata(self):
        """
        Fetch all entries from 'model_date' collection, including their metadatas.

        Returns:
            dict: Raw result from the DB (expected to have a 'results' list of dicts).
        """
        try:
            if hasattr(self.db_client, "get") and asyncio.iscoroutinefunction(self.db_client.get):
                loop = asyncio.get_event_loop()
                coro = self.db_client.get(collection_name="model_date", include=["metadatas"])
                return loop.run_until_complete(coro)
            else:
                return self.db_client.get(collection_name="model_date", include=["metadatas"])
        except Exception:
            return {}

    def _filter_accessible_model_ids(self, all_models, user_id):
        """
        From raw 'model_date' entries, return a set of model_ids the user can 'view'.

        Args:
            all_models: Dict containing 'results': list of model entries.
            user_id: The ID of the user.

        Returns:
            set of str: model_id strings that the user can view.
        """
        accessible_ids = set()
        if not isinstance(all_models, dict) or "results" not in all_models:
            return accessible_ids

        for entry in all_models["results"]:
            metadata = entry.get("metadata", {})
            model_id = metadata.get("model_id")
            if not model_id:
                continue

            # Re‐use check_access by wrapping metadata in a pseudo‐document
            pseudo_doc = {"metadata": metadata}
            if self.check_access(pseudo_doc, user_id, "view"):
                accessible_ids.add(model_id)

        return accessible_ids

    def _consolidate_model_info(self, model_id):
        """
        Build a consolidated metadata dict for a single model_id by querying related tables.

        Args:
            model_id: The model ID to fetch info for.

        Returns:
            dict: Consolidated metadata for that model, or None on error.
        """
        metadata_tables = [
            "model_architectures",
            "model_frameworks",
            "model_datasets",
            "model_training_configs",
            "model_file",
            "model_git",
            "model_images_folder"
        ]

        consolidated = {"model_id": model_id}

        # Fetch and merge metadata from each table
        for table in metadata_tables:
            try:
                if hasattr(self.db_client, "get") and asyncio.iscoroutinefunction(self.db_client.get):
                    loop = asyncio.get_event_loop()
                    coro = self.db_client.get(
                        collection_name=table,
                        where={"model_id": {"$eq": model_id}},
                        include=["metadatas"]
                    )
                    results = loop.run_until_complete(coro)
                else:
                    results = self.db_client.get(
                        collection_name=table,
                        where={"model_id": {"$eq": model_id}},
                        include=["metadatas"]
                    )
            except Exception:
                continue

            if (
                isinstance(results, dict)
                and "results" in results
                and results["results"]
                and "metadata" in results["results"][0]
            ):
                meta = results["results"][0]["metadata"]
                # If there's a 'description', exclude it from the consolidated dict
                if "description" in meta:
                    for key, val in meta.items():
                        if key != "description":
                            consolidated[key] = val
                else:
                    consolidated.update(meta)

        # If 'file' and 'images_folder' keys exist, parse them for structured fields
        file_raw = consolidated.get("file")
        images_folder_raw = consolidated.get("images_folder")

        if file_raw and images_folder_raw:
            try:
                file_info = json.loads(file_raw)
                images_info = json.loads(images_folder_raw)

                model_info = {
                    "model_id": model_id,
                    "creation_date": file_info.get("creation_date"),
                    "last_modified_date": file_info.get("last_modified_date"),
                    "total_chunks": consolidated.get("total_chunks"),
                    "absolute_path": file_info.get("absolute_path"),
                    "images_folder": images_info.get("name"),
                }
                # Append optional fields if present
                for opt_field in ("framework", "version"):
                    if opt_field in consolidated:
                        model_info[opt_field] = consolidated[opt_field]

                return model_info
            except Exception:
                pass

        # If 'file' / 'images_folder' not present or parsing fails, return a simpler dict
        basic_info = {"model_id": model_id}
        for key, val in consolidated.items():
            if key not in ("file", "description", "access_control"):
                basic_info[key] = val

        return basic_info

    # ────────────────────────────────────────────────────────────────────────────────
    # Private: get_accessible_images helpers
    # ────────────────────────────────────────────────────────────────────────────────

    def _fetch_all_images(self):
        """
        Fetch all entries from 'generated_images' collection, including their metadatas.

        Returns:
            dict: Raw result from the DB (expected to have a 'results' list).
        """
        try:
            if hasattr(self.db_client, "get") and asyncio.iscoroutinefunction(self.db_client.get):
                loop = asyncio.get_event_loop()
                coro = self.db_client.get(collection_name="generated_images", include=["metadatas"]
            )
                return loop.run_until_complete(coro)
            else:
                return self.db_client.get(collection_name="generated_images", include=["metadatas"])
        except Exception:
            return {}

    def _filter_accessible_images(self, all_images, user_id):
        """
        From raw 'generated_images' entries, return a list of those the user can 'view'.

        Args:
            all_images: Dict containing 'results': list of image entries.
            user_id: The ID of the user.

        Returns:
            list of dict: Each dict has at least keys: 'id', 'prompt', 'filepath', plus any extra metadata.
        """
        accessible = []
        if not isinstance(all_images, dict) or "results" not in all_images:
            return accessible

        for entry in all_images["results"]:
            # Wrap each entry in a pseudo‐document to use check_access
            if self.check_access(entry, user_id, "view"):
                img_meta = entry.get("metadata", {})
                info = {
                    "id": entry.get("id", ""),
                    "prompt": img_meta.get("prompt", "No prompt"),
                    "filepath": img_meta.get("image_path", "No path")
                }
                # Optional fields
                if "style_tags" in img_meta:
                    info["style_tags"] = img_meta["style_tags"]
                if "clip_score" in img_meta:
                    info["clip_score"] = img_meta["clip_score"]

                accessible.append(info)

        return accessible