import json
import os
import tempfile
import unittest

from jsonschema import ValidationError

from src.data_processing.document_processor.schema_validator import SchemaValidator


class TestSchemaValidator(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()

        # Create a sample schema registry
        self.registry = {
            "registry_name": "Test Schema Registry",
            "schemas": [
                {
                    "schema_id": "test_schema",
                    "schema_version": "1.0",
                    "schema_definition": {
                        "$schema": "http://json-schema.org/draft-07/schema#",
                        "type": "object",
                        "required": ["name", "age"],
                        "properties": {
                            "name": {"type": "string"},
                            "age": {"type": "integer"},
                            "email": {"type": "string", "format": "email"}
                        }
                    },
                    "description": "Test schema v1.0",
                    "updated_date": "2023-01-01"
                },
                {
                    "schema_id": "test_schema",
                    "schema_version": "1.1",
                    "schema_definition": {
                        "$schema": "http://json-schema.org/draft-07/schema#",
                        "type": "object",
                        "required": ["name", "age", "email"],
                        "properties": {
                            "name": {"type": "string"},
                            "age": {"type": "integer"},
                            "email": {"type": "string", "format": "email"},
                            "address": {"type": "string", "default": "N/A"}
                        }
                    },
                    "description": "Test schema v1.1",
                    "updated_date": "2023-02-01"
                },
                {
                    "schema_id": "another_schema",
                    "schema_version": "1.0",
                    "schema_definition": {
                        "$schema": "http://json-schema.org/draft-07/schema#",
                        "type": "object",
                        "required": ["title", "content"],
                        "properties": {
                            "title": {"type": "string"},
                            "content": {"type": "string"},
                            "tags": {"type": "array", "items": {"type": "string"}}
                        }
                    },
                    "description": "Another test schema",
                    "updated_date": "2023-01-15"
                }
            ]
        }

        # Write the registry to a file
        self.registry_path = os.path.join(self.temp_dir.name, "schema_registry.json")
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f)

        # Create the SchemaValidator instance
        self.validator = SchemaValidator(self.registry_path)

    def tearDown(self):
        # Clean up the temporary directory
        self.temp_dir.cleanup()

    def test_init_and_load_schemas(self):
        # Check if schemas were loaded correctly
        self.assertIn("test_schema", self.validator.schemas)
        self.assertIn("another_schema", self.validator.schemas)

        # Check if versions were loaded correctly
        self.assertIn("1.0", self.validator.schemas["test_schema"])
        self.assertIn("1.1", self.validator.schemas["test_schema"])
        self.assertIn("1.0", self.validator.schemas["another_schema"])

        # Check schema content
        self.assertEqual(
            self.validator.schemas["test_schema"]["1.0"]["description"],
            "Test schema v1.0"
        )

    def test_validate_valid_document(self):
        # Create a valid document
        document = {
            "name": "John Doe",
            "age": 30,
            "email": "john@example.com",
            "$schema_version": "1.0"
        }

        # Validate the document
        result = self.validator.validate(document, "test_schema")

        # Check if validation passed
        self.assertTrue(result["valid"])
        self.assertIsNone(result["errors"])
        self.assertEqual(result["document"], document)

    def test_validate_invalid_document(self):
        # Create an invalid document (missing required field)
        document = {
            "name": "John Doe",
            # Missing 'age' field
            "email": "john@example.com",
            "$schema_version": "1.0"
        }

        # Validate the document
        result = self.validator.validate(document, "test_schema")

        # Check if validation failed
        self.assertFalse(result["valid"])
        self.assertIsNotNone(result["errors"])

    def test_validate_without_schema_version(self):
        # Create a valid document without specifying schema version
        document = {
            "name": "John Doe",
            "age": 30,
            "email": "john@example.com"
        }

        # Validate the document
        result = self.validator.validate(document, "test_schema")

        # Check if validation passed and latest version was used
        self.assertTrue(result["valid"])
        self.assertEqual(result["document"]["$schema_version"], "1.1")

    def test_get_schema(self):
        # Get schema with specific version
        schema = self.validator.get_schema("test_schema", "1.0")

        # Check schema content
        self.assertEqual(schema["schema_id"], "test_schema")
        self.assertEqual(schema["schema_version"], "1.0")
        self.assertEqual(schema["description"], "Test schema v1.0")

        # Get schema with latest version
        schema = self.validator.get_schema("test_schema")

        # Should return the latest version (1.1)
        self.assertEqual(schema["schema_version"], "1.1")

    def test_list_schemas(self):
        # Get list of schemas
        schemas = self.validator.list_schemas()

        # Check if all schemas are listed
        self.assertEqual(len(schemas), 2)  # two schema IDs

        # Check schema details
        schema_ids = [schema["schema_id"] for schema in schemas]
        self.assertIn("test_schema", schema_ids)
        self.assertIn("another_schema", schema_ids)

        # Find test_schema in the list
        test_schema = next(s for s in schemas if s["schema_id"] == "test_schema")
        self.assertEqual(test_schema["latest_version"], "1.1")
        self.assertEqual(test_schema["all_versions"], ["1.0", "1.1"])

    def test_get_latest_version(self):
        # Check latest version for test_schema (should be 1.1)
        version = self.validator._get_latest_version("test_schema")
        self.assertEqual(version, "1.1")

        # Check latest version for another_schema (should be 1.0)
        version = self.validator._get_latest_version("another_schema")
        self.assertEqual(version, "1.0")

    def test_validate_schema_compatibility(self):
        # Create a document valid for v1.0 but missing required field for v1.1
        document = {
            "name": "John Doe",
            "age": 30,
            # Missing 'email' field which is required in v1.1
            "$schema_version": "1.0"
        }

        # Check compatibility with v1.1
        issues = self.validator.validate_schema_compatibility(document, "test_schema", "1.1")

        # Should have compatibility issues
        self.assertTrue(len(issues) > 0)
        self.assertIn("Missing required field: email", issues)

        # Add the missing field
        document["email"] = "john@example.com"

        # Check compatibility again
        issues = self.validator.validate_schema_compatibility(document, "test_schema", "1.1")

        # Should be compatible now
        self.assertEqual(len(issues), 0)

    def test_migrate_document(self):
        # Create a document valid for v1.0
        document = {
            "name": "John Doe",
            "age": 30,
            "email": "john@example.com",
            "$schema_version": "1.0"
        }

        # Migrate to v1.1
        result = self.validator.migrate_document(document, "test_schema", "1.1")

        # Check migration result
        self.assertTrue(result["valid"])
        self.assertEqual(result["document"]["$schema_version"], "1.1")
        self.assertEqual(result["document"]["address"], "N/A")  # Default value added

    def test_migrate_incompatible_document(self):
        # Create a document missing required fields for v1.1
        document = {
            "name": "John Doe",
            "age": 30,
            # Missing 'email' field which is required in v1.1
            "$schema_version": "1.0"
        }

        # Attempt to migrate to v1.1
        with self.assertRaises(ValidationError):
            self.validator.migrate_document(document, "test_schema", "1.1")

    def test_error_handling(self):
        # Test with non-existent registry file
        with self.assertRaises(FileNotFoundError):
            SchemaValidator("non_existent_file.json")

        # Test with invalid schema ID
        with self.assertRaises(ValueError):
            self.validator.validate({"name": "John"}, "non_existent_schema")

        # Test with invalid schema version
        document = {
            "name": "John",
            "$schema_version": "999.999"  # Non-existent version
        }
        with self.assertRaises(ValueError):
            self.validator.validate(document, "test_schema")

    def test_invalid_registry_format(self):
        # Create a temporary file with invalid JSON
        invalid_registry_path = os.path.join(self.temp_dir.name, "invalid_registry.json")
        with open(invalid_registry_path, 'w') as f:
            f.write("{ invalid_json: }")

        # Try to create a validator with the invalid registry
        with self.assertRaises(ValueError):
            SchemaValidator(invalid_registry_path)

        # Create a temporary file with valid JSON but invalid registry structure
        invalid_structure_path = os.path.join(self.temp_dir.name, "invalid_structure.json")
        with open(invalid_structure_path, 'w') as f:
            json.dump({"not_a_valid_registry": True}, f)

        # Try to create a validator with the invalid structure
        with self.assertRaises(ValueError):
            SchemaValidator(invalid_structure_path)


if __name__ == '__main__':
    unittest.main()