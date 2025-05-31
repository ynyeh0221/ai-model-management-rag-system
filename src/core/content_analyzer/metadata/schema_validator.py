"""
SchemaValidator for the AI Model Management RAG System.

This module provides functionality to validate documents against schema definitions
stored in a schema registry. It handles schema loading, validation, and management
of schema versions.
"""

import json
import logging
import os
from typing import Dict, Any, List, Optional

import jsonschema
from jsonschema import validate, ValidationError, SchemaError

logger = logging.getLogger(__name__)

schema_version_str = "$schema_version"

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
        self._load_schemas()
    
    def _load_schemas(self):
        """
        Load schemas from the registry.
        
        Reads the schema registry JSON file and load all schema definitions into memory.
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
            document: The document to validate.
            schema_id: The ID of the schema to validate against.

        Returns:
            A dictionary with keys:
                - "valid": True if the document is valid, False otherwise.
                - "document": The validated document (or the document as modified).
                - "errors": An error message if validation fails, otherwise None.

        Raises:
            ValueError: If the schema ID is not found or if the document doesn't specify a schema version.
        """
        if schema_id not in self.schemas:
            raise ValueError(f"Schema ID '{schema_id}' not found in registry")

        # Determine which schema version to use.
        if schema_version_str in document:
            requested_version = document[schema_version_str]
            if requested_version not in self.schemas[schema_id]:
                raise ValueError(f"Schema version '{requested_version}' not found for schema ID '{schema_id}'")
            schema_version = requested_version
        else:
            # Default to the latest version if not specified
            schema_version = self._get_latest_version(schema_id)
            document = document.copy()  # Don't modify the original
            document[schema_version_str] = schema_version
            logger.debug(f"No schema version specified, using latest version: {schema_version}")

        # Get schema definition
        schema_def = self.schemas[schema_id][schema_version]["definition"]

        try:
            validate(instance=document, schema=schema_def)
        except ValidationError as e:
            error_msg = f"Document validation failed for schema '{schema_id}' version '{schema_version}': {e.message}"
            if e.path:
                error_msg += f" at path: {'.'.join(str(p) for p in e.path)}"
            logger.error(error_msg)
            return {"valid": False, "errors": error_msg, "document": document}

        # If validation passed, return a dictionary indicating success.
        return {"valid": True, "errors": None, "document": document}

    def get_schema(self, schema_id: str, version: Optional[str] = None) -> Dict[str, Any]:
        """
        Get a schema by ID and optionally version.
        
        Args:
            schema_id: The ID of the schema to retrieve
            version: The version of the schema to retrieve (default to latest)
            
        Returns:
            The schema definition
            
        Raises:
            ValueError: If the schema ID or version is not found
        """
        if schema_id not in self.schemas:
            raise ValueError(f"Schema ID '{schema_id}' not found in registry")
        
        # If a version is not specified, use the latest
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
    
    def validate_schema_compatibility(
        self,
        document: Dict[str, Any],
        target_schema_id: str,
        target_version: str
    ) -> List[str]:
        """
        Validate if a document is compatible with a different schema version.

        Args:
            document: The document to validate
            target_schema_id: The target schema ID
            target_version: The target schema version

        Returns:
            List of compatibility issues, empty if fully compatible

        Raises:
            ValueError: If the target schema or version is not found, or if the document
                        does not specify a schema version
        """
        # 1) Ensure the target schema ID and version exist
        self._ensure_schema_exists(target_schema_id, target_version)

        # 2) Extract the document’s current version, raising if missing
        current_version = document.get(schema_version_str)
        if not current_version:
            raise ValueError("Document does not specify a schema version")

        # 3) Retrieve the target schema’s definition
        target_schema = self.schemas[target_schema_id][target_version]["definition"]

        # 4) Aggregate compatibility issues
        issues: List[str] = []
        issues.extend(self._collect_missing_required_fields(document, target_schema))
        issues.extend(self._collect_type_mismatches(document, target_schema))

        return issues

    def _ensure_schema_exists(self, schema_id: str, version: str) -> None:
        """
        Raise a ValueError if schema_id or version is not present in self.schemas.
        """
        if schema_id not in self.schemas:
            raise ValueError(f"Target schema ID '{schema_id}' not found in registry")
        if version not in self.schemas[schema_id]:
            raise ValueError(
                f"Target schema version '{version}' not found for schema ID '{schema_id}'"
            )

    def _collect_missing_required_fields(
        self,
        document: Dict[str, Any],
        target_schema: Dict[str, Any]
    ) -> List[str]:
        """
        Return a list of "Missing required field: <field>" messages for any field
        listed in target_schema["required"] but not present in a document.
        Ignores schema_version_str even if listed as required.
        """
        issues: List[str] = []
        required_fields = target_schema.get("required", [])
        for field_name in required_fields:
            if field_name == schema_version_str:
                continue
            if field_name not in document:
                issues.append(f"Missing required field: {field_name}")
        return issues

    def _collect_type_mismatches(
        self,
        document: Dict[str, Any],
        target_schema: Dict[str, Any]
    ) -> List[str]:
        """
        Return a list of "<Field> should be a <type>" messages for any property
        in target_schema["properties"] where the document’s value does not match
        the declared type.
        Supported types: string, number, integer, boolean, array, object.
        """
        issues: List[str] = []
        properties = target_schema.get("properties", {})

        for prop_name, prop_schema in properties.items():
            if prop_name not in document:
                continue  # Nothing to validate if the document doesn’t define this property

            # Only validate if a type is specified in the schema
            expected_type = prop_schema.get("type")
            if not expected_type:
                continue

            actual_value = document[prop_name]
            if not self._is_value_compatible(actual_value, expected_type):
                issues.append(f"Field '{prop_name}' should be a {expected_type}")

        return issues

    def _is_value_compatible(self, value: Any, expected_type: str) -> bool:
        """
        Return True if `value` matches the JSON Schema type `expected_type`.
        Supported expected_type strings: "string", "number", "integer",
        "boolean", "array", "object".
        """
        if expected_type == "string":
            return isinstance(value, str)
        if expected_type == "number":
            return isinstance(value, (int, float))
        if expected_type == "integer":
            return isinstance(value, int)
        if expected_type == "boolean":
            return isinstance(value, bool)
        if expected_type == "array":
            return isinstance(value, list)
        if expected_type == "object":
            return isinstance(value, dict)
        # If an unrecognized type appears, treat as incompatible
        return False
    
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
        migrated[schema_version_str] = target_version
        
        # Get target schema
        target_schema = self.schemas[target_schema_id][target_version]["definition"]
        
        # Add default values for missing fields
        if "properties" in target_schema:
            for prop_name, prop_schema in target_schema["properties"].items():
                if prop_name not in migrated and "default" in prop_schema:
                    migrated[prop_name] = prop_schema["default"]
        
        # Validate the migrated document
        return self.validate(migrated, target_schema_id)
