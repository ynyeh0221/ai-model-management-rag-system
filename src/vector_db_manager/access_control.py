class AccessControlManager:
    def __init__(self, db_client=None):
        """
        Initialize the AccessControlManager with a database client.
        
        Args:
            db_client: A client connection to the database that stores access control information.
                      This could be Chroma or an additional metadata database.
        """
        self.db_client = db_client
        # Define permission hierarchy - higher levels include lower levels
        self.permission_hierarchy = {
            "owner": ["view", "edit", "share", "delete"],
            "edit": ["view", "edit"],
            "share": ["view", "share"],
            "view": ["view"]
        }
    
    def check_access(self, document, user_id, permission_type="view"):
        """
        Check if a user has access to a document.
        
        Args:
            document: Document object or dict containing access_control metadata
            user_id: The ID of the user requesting access
            permission_type: Type of permission to check ("view", "edit", "share", "delete")
            
        Returns:
            bool: True if user has the required permission, False otherwise
        """
        if not document or "metadata" not in document or "access_control" not in document["metadata"]:
            # If document has no access control metadata, deny access by default
            return False
        
        access_control = document["metadata"]["access_control"]
        
        # Owner has all permissions
        if access_control.get("owner") == user_id:
            return True
        
        # Check if user is explicitly mentioned in the specific permission list
        if permission_type in access_control:
            permission_list = access_control.get(f"{permission_type}_permissions", [])
            if user_id in permission_list:
                return True
        
        # Check group permissions - check all permission types that would include the requested permission
        for higher_perm, included_perms in self.permission_hierarchy.items():
            if permission_type in included_perms:
                group_list = access_control.get(f"{higher_perm}_permissions", [])
                
                # Check if any of the user's groups are in the permission list
                user_groups = self._get_user_groups(user_id)
                if any(group in group_list for group in user_groups):
                    return True
        
        # Check for public access for view permission
        if permission_type == "view" and "public" in access_control.get("view_permissions", []):
            return True
            
        return False
    
    def grant_access(self, document_id, user_id, permission_type="view"):
        """
        Grant access to a document for a user.
        
        Args:
            document_id: The ID of the document
            user_id: The ID of the user to grant access to
            permission_type: Type of permission to grant ("view", "edit", "share", "delete")
            
        Returns:
            bool: True if access was granted successfully, False otherwise
        """
        if not self.db_client:
            raise ValueError("Database client is required to modify access control")
            
        # Retrieve the document
        document = self._get_document(document_id)
        if not document:
            return False
            
        # Get current access control or initialize if not exists
        metadata = document.get("metadata", {})
        access_control = metadata.get("access_control", {
            "owner": None,
            "view_permissions": [],
            "edit_permissions": [],
            "share_permissions": []
        })
        
        # Update the appropriate permission list if user not already in it
        perm_list_key = f"{permission_type}_permissions"
        if perm_list_key in access_control:
            if user_id not in access_control[perm_list_key]:
                access_control[perm_list_key].append(user_id)
        else:
            access_control[perm_list_key] = [user_id]
            
        # Update the document's metadata
        metadata["access_control"] = access_control
        return self._update_document_metadata(document_id, metadata)
    
    def revoke_access(self, document_id, user_id, permission_type="view"):
        """
        Revoke access to a document for a user.
        
        Args:
            document_id: The ID of the document
            user_id: The ID of the user to revoke access from
            permission_type: Type of permission to revoke ("view", "edit", "share", "delete")
            
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
        access_control = metadata.get("access_control", {})
        
        # Cannot revoke owner permissions
        if permission_type == "owner" and access_control.get("owner") == user_id:
            return False
            
        # Remove user from the permission list
        perm_list_key = f"{permission_type}_permissions"
        if perm_list_key in access_control and user_id in access_control[perm_list_key]:
            access_control[perm_list_key].remove(user_id)
            
        # Update the document's metadata
        metadata["access_control"] = access_control
        return self._update_document_metadata(document_id, metadata)
    
    def create_access_filter(self, user_id):
        """
        Create a filter for access control in queries.
        This generates a filter that can be applied to Chroma queries to only return
        documents the user has permission to view.
        
        Args:
            user_id: The ID of the user making the query
            
        Returns:
            dict: A filter dictionary that can be passed to Chroma's query method
        """
        user_groups = self._get_user_groups(user_id)
        
        # Create a filter that matches documents where:
        # 1. The user is the owner
        # 2. The user is explicitly in view_permissions
        # 3. Any of the user's groups are in view_permissions
        # 4. The document is public (has "public" in view_permissions)
        
        filter_dict = {
            "$or": [
                {"metadata.access_control.owner": user_id},
                {"metadata.access_control.view_permissions": {"$contains": user_id}},
                {"metadata.access_control.edit_permissions": {"$contains": user_id}},
                {"metadata.access_control.share_permissions": {"$contains": user_id}},
                {"metadata.access_control.view_permissions": {"$contains": "public"}}
            ]
        }
        
        # Add group permissions if user belongs to any groups
        if user_groups:
            for group in user_groups:
                filter_dict["$or"].extend([
                    {"metadata.access_control.view_permissions": {"$contains": group}},
                    {"metadata.access_control.edit_permissions": {"$contains": group}},
                    {"metadata.access_control.share_permissions": {"$contains": group}}
                ])
                
        return filter_dict
    
    def transfer_ownership(self, document_id, current_owner_id, new_owner_id):
        """
        Transfer ownership of a document from one user to another.
        
        Args:
            document_id: The ID of the document
            current_owner_id: The ID of the current owner
            new_owner_id: The ID of the new owner
            
        Returns:
            bool: True if ownership was transferred successfully, False otherwise
        """
        if not self.db_client:
            raise ValueError("Database client is required to modify access control")
            
        # Retrieve the document
        document = self._get_document(document_id)
        if not document:
            return False
            
        # Verify current ownership
        metadata = document.get("metadata", {})
        access_control = metadata.get("access_control", {})
        
        if access_control.get("owner") != current_owner_id:
            return False  # Only the current owner can transfer ownership
            
        # Update owner
        access_control["owner"] = new_owner_id
        
        # Make sure the new owner has all permissions
        for perm_type in ["view", "edit", "share"]:
            perm_key = f"{perm_type}_permissions"
            if perm_key in access_control and new_owner_id not in access_control[perm_key]:
                access_control[perm_key].append(new_owner_id)
                
        # Update the document's metadata
        metadata["access_control"] = access_control
        return self._update_document_metadata(document_id, metadata)
    
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
        
        # This could be implemented by:
        # 1. Querying an identity provider (e.g., LDAP, Okta)
        # 2. Checking a local database of group memberships
        # 3. Using a cache for performance
        
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
            self.db_client.update(
                ids=[document_id],
                metadatas=[metadata]
            )
            return True
        except Exception as e:
            print(f"Error updating document metadata: {e}")
            return False
    
    def log_access_attempt(self, document_id, user_id, permission_type, success):
        """
        Log an access attempt for security auditing.
        
        Args:
            document_id: The ID of the document
            user_id: The ID of the user
            permission_type: Type of permission requested
            success: Whether access was granted
            
        Returns:
            bool: True if log was recorded successfully, False otherwise
        """
        # In a real implementation, this would write to a secure audit log
        # Could use a separate database table or logging service
        
        log_entry = {
            "timestamp": self._get_current_timestamp(),
            "document_id": document_id,
            "user_id": user_id,
            "permission_type": permission_type,
            "success": success,
            "ip_address": self._get_user_ip(user_id)
        }
        
        # Example implementation - adapt based on actual logging mechanism
        try:
            if hasattr(self.db_client, "log_access"):
                self.db_client.log_access(log_entry)
            return True
        except Exception as e:
            print(f"Error logging access attempt: {e}")
            return False
    
    def _get_current_timestamp(self):
        """Get the current timestamp in ISO format."""
        from datetime import datetime
        return datetime.utcnow().isoformat()
    
    def _get_user_ip(self, user_id):
        """Get the IP address of the user's current session."""
        # This would be implemented based on the application's session management
        return "0.0.0.0"  # Placeholder

    def get_document_permissions(self, document: dict) -> dict:
        """
        Determine and return the access control metadata for the document.

        This simple implementation returns a default permission,
        granting public view rights. In a more complete system, you might
        check the document's metadata, the current user's role, and other parameters.

        Args:
            document (dict): The document for which to determine permissions.

        Returns:
            dict: A dictionary representing access control permissions.
                  For example, {"view": ["public"], "edit": []}
        """
        # For demonstration, we return default permissions.
        return {"view": ["public"], "edit": []}
