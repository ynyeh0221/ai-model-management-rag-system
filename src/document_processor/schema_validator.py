"""
SchemaValidator for the AI Model Management RAG System.

This module provides functionality to validate documents against schema definitions
stored in a schema registry. It handles schema loading, validation, and management
of schema versions.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import jsonschema
from jsonschema import validate, ValidationError, SchemaError
from pydantic import BaseModel, Field, validator, root_validator

logger = logging.getLogger(__name__)

class SchemaValidator:
    """
    Validates documents against schema definitions from a registry.
    
    This class loads schema definitions from a JSON registry file and provides methods
    to validate documents against these schemas. It supports schema versioning and
    can handle backward compatibility checks.
    
    Attributes:
        schema_registry_path: Path to the schema registry JSON file
        schemas: Dictionary of loaded schemas, keyed by schema_id
    """
    
    def __init__(self, schema_registry_path: str):
        """
        Initialize the SchemaValidator.
        
        Args:
            schema_registry_path: Path to the schema registry JSON file
        
        Raises:
            FileNotFoundError: If the schema registry file doesn't exist
            ValueError: If the schema registry is invalid
        """
        self.schema_registry_path = schema_registry_path
        self.schemas = {}
        # self._load_schemas()
    
    def _load_schemas(self):
        """
        Load schemas from the registry.
        
        Reads the schema registry JSON file and loads all schema definitions into memory.
        Organizes schemas by ID and version for efficient lookup.
        
        Raises:
            FileNotFoundError: If the schema registry file doesn't exist
            ValueError: If the schema registry is invalid JSON or missing required fields
            SchemaError: If any schema definition is invalid
        """
        if not os.path.exists(self.schema_registry_path):
            raise FileNotFoundError(f"Schema registry not found at: {self.schema_registry_path}")
        
        try:
            with open(self.schema_registry_path, 'r', encoding='utf-8') as f:
                registry = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in schema registry: {e}")
        
        # Validate registry structure
        if not isinstance(registry, dict):
            raise ValueError("Schema registry must be a JSON object")
        
        if "registry_name" not in registry:
            raise ValueError("Schema registry must have a 'registry_name' field")
        
        if "schemas" not in registry or not isinstance(registry["schemas"], list):
            raise ValueError("Schema registry must have a 'schemas' array")
        
        # Process each schema
        for schema_entry in registry["schemas"]:
            # Validate schema entry
            if not isinstance(schema_entry, dict):
                logger.warning("Invalid schema entry (not an object), skipping")
                continue
            
            required_fields = ["schema_id", "schema_version", "schema_definition"]
            if not all(field in schema_entry for field in required_fields):
                logger.warning(f"Schema entry missing required fields {required_fields}, skipping")
                continue
            
            schema_id = schema_entry["schema_id"]
            schema_version = schema_entry["schema_version"]
            schema_definition = schema_entry["schema_definition"]
            
            # Validate schema definition is a valid JSON Schema
            try:
                # This will raise SchemaError if the schema is invalid
                jsonschema.Draft7Validator.check_schema(schema_definition)
            except SchemaError as e:
                logger.error(f"Invalid schema definition for {schema_id}: {e}")
                continue
            
            # Store schema by ID and version
            if schema_id not in self.schemas:
                self.schemas[schema_id] = {}
            
            self.schemas[schema_id][schema_version] = {
                "definition": schema_definition,
                "description": schema_entry.get("description", ""),
                "updated_date": schema_entry.get("updated_date")
            }
            
            logger.info(f"Loaded schema: {schema_id} (version {schema_version})")
        
        logger.info(f"Loaded {len(self.schemas)} schema types from registry")
    
    def validate(self, document: Dict[str, Any], schema_id: str) -> Dict[str, Any]:
        """
        Validate a document against a schema.
        
        Args:
            document: The document to validate
            schema_id: The ID of the schema to validate against
            
        Returns:
            The validated document (possibly with defaults applied)
            
        Raises:
            ValueError: If the schema ID is not found or if the document doesn't specify a schema version
            ValidationError: If the document fails validation
        """
        if schema_id not in self.schemas:
            raise ValueError(f"Schema ID '{schema_id}' not found in registry")
        
        # Determine which schema version to use
        if "$schema_version" in document:
            requested_version = document["$schema_version"]
            if requested_version not in self.schemas[schema_id]:
                raise ValueError(f"Schema version '{requested_version}' not found for schema ID '{schema_id}'")
            schema_version = requested_version
        else:
            # Default to latest version if not specified
            schema_version = self._get_latest_version(schema_id)
            
            # Add schema version to document for future reference
            document = document.copy()  # Don't modify the original
            document["$schema_version"] = schema_version
            
            logger.debug(f"No schema version specified, using latest version: {schema_version}")
        
        # Get schema definition
        schema_def = self.schemas[schema_id][schema_version]["definition"]
        
        # Validate document against schema
        try:
            validate(instance=document, schema=schema_def)
        except ValidationError as e:
            # Add context to the error message
            error_msg = f"Document validation failed for schema '{schema_id}' version '{schema_version}': {e.message}"
            if e.path:
                error_msg += f" at path: {'.'.join(str(p) for p in e.path)}"
            
            logger.error(error_msg)
            raise ValidationError(error_msg)
        
        # Return the validated document
        return document
    
    def get_schema(self, schema_id: str, version: Optional[str] = None) -> Dict[str, Any]:
        """
        Get a schema by ID and optionally version.
        
        Args:
            schema_id: The ID of the schema to retrieve
            version: The version of the schema to retrieve (defaults to latest)
            
        Returns:
            The schema definition
            
        Raises:
            ValueError: If the schema ID or version is not found
        """
        if schema_id not in self.schemas:
            raise ValueError(f"Schema ID '{schema_id}' not found in registry")
        
        # If version is not specified, use the latest
        if version is None:
            version = self._get_latest_version(schema_id)
        elif version not in self.schemas[schema_id]:
            raise ValueError(f"Schema version '{version}' not found for schema ID '{schema_id}'")
        
        return {
            "schema_id": schema_id,
            "schema_version": version,
            "definition": self.schemas[schema_id][version]["definition"],
            "description": self.schemas[schema_id][version]["description"],
            "updated_date": self.schemas[schema_id][version]["updated_date"]
        }
    
    def list_schemas(self) -> List[Dict[str, Any]]:
        """
        List all available schemas.
        
        Returns:
            List of schema metadata (ID, versions, description, etc.)
        """
        result = []
        
        for schema_id, versions in self.schemas.items():
            # Get the latest version
            latest_version = self._get_latest_version(schema_id)
            
            # Get schema metadata
            schema_info = {
                "schema_id": schema_id,
                "latest_version": latest_version,
                "description": versions[latest_version]["description"],
                "updated_date": versions[latest_version]["updated_date"],
                "all_versions": sorted(list(versions.keys()), 
                                       key=lambda v: [int(x) for x in v.split('.')])
            }
            
            result.append(schema_info)
        
        # Sort by schema ID
        result.sort(key=lambda x: x["schema_id"])
        
        return result
    
    def _get_latest_version(self, schema_id: str) -> str:
        """
        Get the latest version of a schema.
        
        Args:
            schema_id: The ID of the schema
            
        Returns:
            The latest version string
            
        Raises:
            ValueError: If the schema ID is not found
        """
        if schema_id not in self.schemas:
            raise ValueError(f"Schema ID '{schema_id}' not found in registry")
        
        # Sort versions using semantic versioning
        versions = list(self.schemas[schema_id].keys())
        
        # Parse versions and sort numerically
        def version_key(v):
            return [int(x) for x in v.split('.')]
        
        versions.sort(key=version_key, reverse=True)
        
        return versions[0]
    
    def validate_schema_compatibility(self, document: Dict[str, Any], 
                                     target_schema_id: str, 
                                     target_version: str) -> List[str]:
        """
        Validate if a document is compatible with a different schema version.
        
        This is useful for checking if a document can be migrated to a newer schema version.
        
        Args:
            document: The document to validate
            target_schema_id: The target schema ID
            target_version: The target schema version
            
        Returns:
            List of compatibility issues, empty if fully compatible
            
        Raises:
            ValueError: If the target schema or version is not found
        """
        if target_schema_id not in self.schemas:
            raise ValueError(f"Target schema ID '{target_schema_id}' not found in registry")
        
        if target_version not in self.schemas[target_schema_id]:
            raise ValueError(f"Target schema version '{target_version}' not found for schema ID '{target_schema_id}'")
        
        # Get current document schema version
        current_version = document.get("$schema_version")
        if not current_version:
            raise ValueError("Document does not specify a schema version")
        
        # Get schema definitions
        target_schema = self.schemas[target_schema_id][target_version]["definition"]
        
        compatibility_issues = []
        
        # Check for required fields in target schema that are missing in document
        if "required" in target_schema:
            for field in target_schema["required"]:
                if field not in document and field != "$schema_version":
                    compatibility_issues.append(f"Missing required field: {field}")
        
        # Check for properties with incompatible types
        if "properties" in target_schema:
            for prop_name, prop_schema in target_schema["properties"].items():
                if prop_name in document:
                    # Check type compatibility
                    if "type" in prop_schema:
                        target_type = prop_schema["type"]
                        
                        # Handle type compatibility
                        value = document[prop_name]
                        
                        # Check if the value type is compatible with the target type
                        if target_type == "string" and not isinstance(value, str):
                            compatibility_issues.append(f"Field '{prop_name}' should be a string")
                        elif target_type == "number" and not isinstance(value, (int, float)):
                            compatibility_issues.append(f"Field '{prop_name}' should be a number")
                        elif target_type == "integer" and not isinstance(value, int):
                            compatibility_issues.append(f"Field '{prop_name}' should be an integer")
                        elif target_type == "boolean" and not isinstance(value, bool):
                            compatibility_issues.append(f"Field '{prop_name}' should be a boolean")
                        elif target_type == "array" and not isinstance(value, list):
                            compatibility_issues.append(f"Field '{prop_name}' should be an array")
                        elif target_type == "object" and not isinstance(value, dict):
                            compatibility_issues.append(f"Field '{prop_name}' should be an object")
        
        return compatibility_issues
    
    def migrate_document(self, document: Dict[str, Any], 
                        target_schema_id: str, 
                        target_version: str) -> Dict[str, Any]:
        """
        Attempt to migrate a document to a new schema version.
        
        Args:
            document: The document to migrate
            target_schema_id: The target schema ID
            target_version: The target schema version
            
        Returns:
            The migrated document
            
        Raises:
            ValueError: If the target schema or version is not found
            ValidationError: If the document cannot be migrated
        """
        # Check compatibility first
        compatibility_issues = self.validate_schema_compatibility(
            document, target_schema_id, target_version
        )
        
        if compatibility_issues:
            raise ValidationError(
                f"Cannot migrate document to schema '{target_schema_id}' version '{target_version}'. "
                f"Issues: {', '.join(compatibility_issues)}"
            )
        
        # Create a copy of the document
        migrated = document.copy()
        
        # Update schema version
        migrated["$schema_version"] = target_version
        
        # Get target schema
        target_schema = self.schemas[target_schema_id][target_version]["definition"]
        
        # Add default values for missing fields
        if "properties" in target_schema:
            for prop_name, prop_schema in target_schema["properties"].items():
                if prop_name not in migrated and "default" in prop_schema:
                    migrated[prop_name] = prop_schema["default"]
        
        # Validate the migrated document
        return self.validate(migrated, target_schema_id)
