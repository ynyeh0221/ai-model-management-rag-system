import ast
import unittest
from unittest.mock import Mock

from src.core.content_analyzer.model_script.visitor.code_summary_visitor import CodeSummaryVisitor


class TestCodeSummaryVisitor(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.mock_parent_generator = Mock()
        self.mock_parent_generator.class_hierarchy = {}
        self.mock_parent_generator.current_class = None
        self.mock_parent_generator.class_layers = {}
        self.mock_parent_generator.function_calls = {}
        self.mock_parent_generator.model_layers = []
        self.mock_parent_generator.used_components = set()
        self.mock_parent_generator.model_connections = {}
        self.mock_parent_generator.layer_order = []
        self.mock_parent_generator.base_dir = "/test/base"
        self.mock_parent_generator._eval_constant = Mock(return_value=None)
        self.mock_parent_generator._determine_folder = Mock(return_value="/test/folder")

        self.lines = []
        self.literal_vars = {}
        self.image_folders = set()
        self.default_paths = set()

        self.visitor = CodeSummaryVisitor(
            self.mock_parent_generator,
            self.lines,
            self.literal_vars,
            self.image_folders,
            self.default_paths
        )

    def test_init(self):
        """Test initialization of CodeSummaryVisitor."""
        self.assertEqual(self.visitor.parent_generator, self.mock_parent_generator)
        self.assertEqual(self.visitor.lines, self.lines)
        self.assertEqual(self.visitor.literal_vars, self.literal_vars)
        self.assertEqual(self.visitor.image_folders, self.image_folders)
        self.assertEqual(self.visitor.default_paths, self.default_paths)
        self.assertEqual(self.visitor.call_counter, {})
        self.assertIsNone(self.visitor.current_function)
        self.assertEqual(self.visitor.function_references, {})

    def test_visit_import(self):
        """Test visit_Import method."""
        node = ast.parse("import os, sys").body[0]
        self.visitor.visit_Import(node)
        self.assertIn("Import: os, sys", self.lines)

    def test_visit_import_from(self):
        """Test visit_ImportFrom method."""
        node = ast.parse("from collections import defaultdict, Counter").body[0]
        self.visitor.visit_ImportFrom(node)
        self.assertIn("From collections import defaultdict, Counter", self.lines)

    def test_visit_import_from_no_module(self):
        """Test visit_ImportFrom with no module specified."""
        # Create a node with module=None (relative import)
        node = ast.ImportFrom(module=None, names=[ast.alias(name='something', asname=None)], level=1)
        self.visitor.visit_ImportFrom(node)
        self.assertIn("From  import something", self.lines)

    def test_visit_class_def_simple(self):
        """Test visit_ClassDef with simple class."""
        code = '''
class TestClass:
    """Test docstring"""
    pass
'''
        node = ast.parse(code).body[0]
        self.visitor.visit_ClassDef(node)

        self.assertIn("\nClass: TestClass (inherits from )", self.lines)
        self.assertIn("  Docstring: Test docstring", self.lines)
        self.assertIn("TestClass", self.mock_parent_generator.class_hierarchy)

    def test_visit_class_def_with_inheritance(self):
        """Test visit_ClassDef with inheritance."""
        code = '''
class NetModel(nn.Module):
    pass
'''
        node = ast.parse(code).body[0]
        self.visitor.visit_ClassDef(node)

        self.assertIn("\nClass: NetModel (inherits from Module)", self.lines)
        self.assertIn("  Neural Network Module detected", self.lines)

    def test_visit_function_def_simple(self):
        """Test visit_FunctionDef with simple function."""
        code = '''
def test_func(a, b=10):
    """Test function"""
    return a + b
'''
        node = ast.parse(code).body[0]
        self.mock_parent_generator._eval_constant.return_value = 10

        self.visitor.visit_FunctionDef(node)

        self.assertIn("\nFunction: test_func(a, b)", self.lines)
        self.assertIn("  Docstring: Test function", self.lines)
        self.assertIn("test_func", self.mock_parent_generator.function_calls)

    def test_visit_function_def_forward_method(self):
        """Test visit_FunctionDef with forward method."""
        code = '''
def forward(self, x):
    return x
'''
        node = ast.parse(code).body[0]
        self.mock_parent_generator.current_class = "TestModel"

        self.visitor.visit_FunctionDef(node)

        self.assertIsNone(self.visitor.current_function)  # Should reset after a visit
        forward_info = self.mock_parent_generator.function_calls.get("forward", {})
        self.assertEqual(forward_info.get("class"), "TestModel")

    def test_visit_call_known_function(self):
        """Test visit_Call with known function."""
        code = "test_func(1, 2, key=3)"
        node = ast.parse(code, mode='eval').body

        # Setup mock function_calls
        self.mock_parent_generator.function_calls["test_func"] = {
            'params': ['a', 'b', 'key'],
            'defaults': {'key': 0}
        }
        self.mock_parent_generator._eval_constant.side_effect = lambda x: getattr(x, 'value', None) if hasattr(x,
                                                                                                               'value') else None

        self.visitor.visit_Call(node)

        # Should create a call entry
        call_entries = [k for k in self.mock_parent_generator.function_calls.keys() if k.startswith("test_func_call_")]
        self.assertGreater(len(call_entries), 0)

    def test_visit_call_component_instantiation(self):
        """Test visit_Call with component instantiation."""
        code = "Linear(10, 5)"
        node = ast.parse(code, mode='eval').body

        self.mock_parent_generator.class_layers = {"Linear": []}

        self.visitor.visit_Call(node)

        self.assertIn("Linear", self.mock_parent_generator.used_components)

    def test_visit_assign_simple(self):
        """Test visit_Assign with simple assignment."""
        code = "batch_size = 32"
        node = ast.parse(code).body[0]
        self.mock_parent_generator._eval_constant.return_value = 32

        self.visitor.visit_Assign(node)

        self.assertEqual(self.literal_vars.get("batch_size"), 32)

    def test_handle_dataloader(self):
        """Test _handle_dataloader method."""
        code = "train_loader = DataLoader(dataset, batch_size=64)"
        node = ast.parse(code).body[0]
        self.mock_parent_generator._eval_constant.return_value = 64

        self.visitor._handle_dataloader(node)

        self.assertEqual(self.literal_vars.get("batch_size"), 64)
        expected_line = "Variable: batch_size = 64 (from DataLoader train_loader)"
        self.assertIn(expected_line, self.lines)

    def test_handle_layer_assignment(self):
        """Test _handle_layer_assignment method."""
        code = "self.linear = nn.Linear(10, 5)"
        node = ast.parse(code).body[0]
        self.mock_parent_generator.current_class = "TestModel"
        self.mock_parent_generator._eval_constant.side_effect = lambda x: getattr(x, 'value',
                                                                                  ast.unparse(x) if hasattr(ast,
                                                                                                            'unparse') else str(
                                                                                      x))

        self.visitor._handle_layer_assignment(node)

        self.assertIn("linear", self.mock_parent_generator.layer_order)
        self.assertGreater(len(self.mock_parent_generator.model_layers), 0)

        layer_info = self.mock_parent_generator.model_layers[-1]
        self.assertEqual(layer_info['name'], 'linear')
        self.assertEqual(layer_info['class'], 'TestModel')

    def test_handle_sequential_assignment(self):
        """Test _handle_sequential_assignment method."""
        code = "self.features = nn.Sequential(nn.Linear(10, 5), nn.ReLU())"
        node = ast.parse(code).body[0]
        self.mock_parent_generator.current_class = "TestModel"

        self.visitor._handle_sequential_assignment(node)

        self.assertIn("features", self.mock_parent_generator.layer_order)
        self.assertGreater(len(self.mock_parent_generator.model_layers), 0)

        layer_info = self.mock_parent_generator.model_layers[-1]
        self.assertEqual(layer_info['name'], 'features')
        self.assertEqual(layer_info['layer_type'], 'Sequential')

    def test_visit_for_range_loop(self):
        """Test visit_For with range loop for epoch detection."""
        code = '''
for epoch in range(100):
    pass
'''
        node = ast.parse(code).body[0]
        self.mock_parent_generator._eval_constant.return_value = 100

        self.visitor.visit_For(node)

        self.assertEqual(self.literal_vars.get("num_epochs"), 100)
        self.assertIn("Variable: num_epochs = 100 (from training loop)", self.lines)

    def test_visit_for_range_with_start_stop(self):
        """Test visit_For with range(start, stop)."""
        code = '''
for i in range(1, 51):
    pass
'''
        node = ast.parse(code).body[0]
        self.mock_parent_generator._eval_constant.side_effect = lambda x: {
            node.iter.args[0]: 1,
            node.iter.args[1]: 51
        }.get(x, None)

        self.visitor.visit_For(node)

        self.assertEqual(self.literal_vars.get("num_epochs"), 51)

    def test_analyze_forward_method(self):
        """Test _analyze_forward_method."""
        code = '''
def forward(self, x):
    x = self.conv1(x)
    x = self.relu(x)
    return self.fc(x)
'''
        node = ast.parse(code).body[0]
        self.mock_parent_generator.current_class = "TestModel"
        self.mock_parent_generator.model_layers = [
            {'name': 'conv1', 'class': 'TestModel', 'layer_type': 'Conv2d'},
            {'name': 'relu', 'class': 'TestModel', 'layer_type': 'ReLU'},
            {'name': 'fc', 'class': 'TestModel', 'layer_type': 'Linear'}
        ]
        self.mock_parent_generator.class_layers = {
            'Conv2d': [], 'ReLU': [], 'Linear': []
        }

        self.visitor._analyze_forward_method(node)

        # Should add references and connections
        self.assertGreater(len(self.mock_parent_generator.used_components), 0)

    def test_process_call_for_references(self):
        """Test _process_call_for_references method."""
        code = "self.layer1(x)"
        node = ast.parse(code, mode='eval').body
        refs = set()

        self.mock_parent_generator.current_class = "TestModel"
        self.mock_parent_generator.model_layers = [
            {'name': 'layer1', 'class': 'TestModel', 'layer_type': 'Linear'}
        ]
        self.mock_parent_generator.class_layers = {'Linear': []}

        self.visitor._process_call_for_references(node, refs)

        self.assertIn('Linear', refs)
        self.assertIn('Linear', self.mock_parent_generator.used_components)

    def test_handle_top_level_model_instantiation(self):
        """Test _handle_top_level_model_instantiation method."""
        code = "model = MyModel()"
        node = ast.parse(code).body[0]
        self.mock_parent_generator.class_layers = {"MyModel": []}

        # Mock ast.unparse if it doesn't exist (Python < 3.9)
        try:
            ast.unparse(node.value)
        except AttributeError:
            # For older Python versions, mock the behavior
            pass

        self.visitor._handle_top_level_model_instantiation(node)

        model_line = next((line for line in self.lines if "model =" in line), None)
        self.assertIsNotNone(model_line)
        self.assertIn("MyModel", self.mock_parent_generator.used_components)

    def test_function_context_tracking(self):
        """Test that function context is properly tracked."""
        code = '''
def outer_func():
    def inner_func():
        pass
    inner_func()
'''
        tree = ast.parse(code)

        # Visit the tree
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                self.visitor.visit_FunctionDef(node)

        # Should have function references
        self.assertIn("outer_func", self.visitor.function_references)
        self.assertIn("inner_func", self.visitor.function_references)

    def test_default_path_detection(self):
        """Test detection of default paths in function parameters."""
        code = '''
def save_model(model, save_dir="./models"):
    pass
'''
        node = ast.parse(code).body[0]
        self.mock_parent_generator._eval_constant.return_value = "./models"
        self.mock_parent_generator._determine_folder.return_value = "/test/models"

        self.visitor.visit_FunctionDef(node)

        self.mock_parent_generator._determine_folder.assert_called()
        self.assertIn("/test/models", self.default_paths)


if __name__ == '__main__':
    # Create a test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCodeSummaryVisitor)

    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print(f"\nTests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")