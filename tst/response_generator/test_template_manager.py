import shutil
import tempfile
import unittest

from src.response_generator.template_manager import TemplateManager


class TestTemplateManager(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test templates
        self.test_dir = tempfile.mkdtemp()
        self.manager = TemplateManager(templates_dir=self.test_dir)

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_save_and_get_template(self):
        template_id = "welcome"
        content = "Hello, {{ name }}!"
        version = self.manager.save_template(template_id, content)

        self.assertEqual(version, "1.0")
        fetched = self.manager.get_template(template_id)
        self.assertEqual(fetched, content)

    def test_render_template(self):
        template_id = "greeting"
        content = "Hi, {{ user }}!"
        self.manager.save_template(template_id, content)

        rendered = self.manager.render_template(template_id, {"user": "Alice"})
        self.assertEqual(rendered, "Hi, Alice!")

    def test_version_incrementing(self):
        template_id = "multi"
        self.manager.save_template(template_id, "Version 1")
        version2 = self.manager.save_template(template_id, "Version 2")

        self.assertEqual(version2, "1.1")
        latest = self.manager.get_template(template_id)
        self.assertEqual(latest, "Version 2")

    def test_template_history(self):
        template_id = "history_test"
        self.manager.save_template(template_id, "First")
        self.manager.save_template(template_id, "Second")

        history = self.manager.get_template_history(template_id)
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["version"], "1.0")
        self.assertEqual(history[1]["version"], "1.1")

    def test_get_specific_version(self):
        template_id = "specific"
        self.manager.save_template(template_id, "v1 content", version="1.0")
        self.manager.save_template(template_id, "v2 content", version="2.0")

        v1 = self.manager.get_template(template_id, version="1.0")
        v2 = self.manager.get_template(template_id, version="2.0")

        self.assertEqual(v1, "v1 content")
        self.assertEqual(v2, "v2 content")

    def test_missing_template(self):
        result = self.manager.get_template("nonexistent")
        self.assertIsNone(result)
        with self.assertRaises(ValueError):
            self.manager.render_template("nonexistent", {})


if __name__ == "__main__":
    unittest.main()
