import unittest

from src.core.colab_generator.code_generator import CodeGenerator


class TestCodeGenerator(unittest.TestCase):

    def setUp(self):
        self.generator = CodeGenerator()

    def test_generate_full_script_no_overlap(self):
        chunks = [
            {"text": "def foo():\n    return 42\n", "offset": 0},
            {"text": "def bar():\n    return 'hello'", "offset": 100}
        ]
        expected = "def foo():\n    return 42\n" \
                   "def bar():\n    return 'hello'"
        result = self.generator.generate_full_script(chunks, overlap=0, use_offset=True)
        self.assertEqual(result.strip(), expected.strip())

    def test_generate_full_script_with_fixed_overlap(self):
        # Two chunks, second chunk repeats the last 10 characters of the first
        base = "def greet():\n    print('Hello')\n    return 'Hi'"
        overlap = 10
        chunk1 = {"text": base, "offset": 0}
        chunk2 = {"text": base[-overlap:] + "\ndef bye():\n    return 'Bye'", "offset": len(base) - overlap}

        chunks = [chunk1, chunk2]

        expected = base + "\n" + "def bye():\n    return 'Bye'"
        result = self.generator.generate_full_script(chunks, overlap=overlap, use_offset=True)
        self.assertEqual(result.strip(), expected.strip())

    def test_generate_full_script_empty(self):
        result = self.generator.generate_full_script([], overlap=200, use_offset=True)
        self.assertEqual(result, "")

    def test_generate_full_script_identical_chunks(self):
        chunk = "def test():\n    pass"
        chunks = [
            {"text": chunk, "offset": 0},
            {"text": chunk, "offset": 0}
        ]
        result = self.generator.generate_full_script(chunks, overlap=len(chunk), use_offset=True)
        expected = chunk
        self.assertIn("def test()", result)
        self.assertEqual(result.strip(), expected.strip())

    def test_generate_full_script_with_offsets(self):
        """Ensure full reconstruction from structured chunks using offset sorting."""
        chunks = [
            {"text": "def a():\n    return 1\n", "offset": 0},
            {"text": "def b():\n    return 2\n", "offset": 25},
            {"text": "def c():\n    return 3\n", "offset": 50},
        ]

        result = self.generator.generate_full_script(chunks, use_offset=True, overlap=0)

        self.assertIn("def a()", result)
        self.assertIn("def b()", result)
        self.assertIn("def c()", result)
        self.assertTrue(result.find("def a()") < result.find("def b()") < result.find("def c()"))


if __name__ == '__main__':
    unittest.main()
