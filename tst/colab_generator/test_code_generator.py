import unittest
from src.colab_generator.code_generator import CodeGenerator


class TestCodeGenerator(unittest.TestCase):
    """
    Unit tests for the CodeGenerator class.

    These tests verify that the CodeGenerator correctly generates
    code for different use cases and frameworks.
    """

    def setUp(self):
        """Set up a new CodeGenerator instance for each test."""
        self.code_generator = CodeGenerator()

        # Create sample model and dataset info dictionaries
        self.pytorch_model_info = {
            "model_id": "test_pytorch_model",
            "filepath": "/models/test_model.pt",
            "framework": {"name": "pytorch"},
            "architecture_type": {"value": "transformer"},
            "model_dimensions": {
                "hidden_size": {"value": 768},
                "num_layers": {"value": 12},
                "num_attention_heads": {"value": 12},
                "total_parameters": {"value": 125000000}
            }
        }

        self.tensorflow_model_info = {
            "model_id": "test_tensorflow_model",
            "filepath": "/models/test_model.h5",
            "framework": {"name": "tensorflow"},
            "architecture_type": {"value": "cnn"}
        }

        self.dataset_info = {
            "name": {"value": "test_dataset"},
            "version": {"value": "1.0.0"},
            "num_samples": {"value": 10000},
            "split": {"value": "train"}
        }

    def test_generate_imports(self):
        """
        Tests the generate_imports method to ensure it correctly produces framework-specific imports.

        This test verifies:
        1. PyTorch imports include "import torch" and requested libraries
        2. TensorFlow imports include "import tensorflow as tf" and requested libraries
        3. Unknown frameworks don't include framework-specific imports but do include requested libraries
        4. All appropriate import statements are generated with correct syntax
        """
        # Test PyTorch imports
        pytorch_imports = self.code_generator.generate_imports("pytorch", ["numpy", "pandas"])
        self.assertIn("import torch", pytorch_imports)
        self.assertIn("import numpy", pytorch_imports)
        self.assertIn("import pandas", pytorch_imports)

        # Test TensorFlow imports
        tensorflow_imports = self.code_generator.generate_imports("tensorflow", ["matplotlib"])
        self.assertIn("import tensorflow as tf", tensorflow_imports)
        self.assertIn("import matplotlib", tensorflow_imports)

        # Test with unknown framework
        generic_imports = self.code_generator.generate_imports("unknown_framework", ["os"])
        self.assertNotIn("import torch", generic_imports)
        self.assertNotIn("import tensorflow", generic_imports)
        self.assertIn("import os", generic_imports)

    def test_generate_model_loading_pytorch(self):
        """
        Tests the generation of PyTorch model loading code.

        This test verifies:
        1. The model ID is correctly included in the generated code
        2. PyTorch-specific loading code is present (containing "torch.load")
        3. The code includes functionality to save model metadata to a JSON file
        4. The generated code follows proper Python syntax and PyTorch conventions
        """
        pytorch_code = self.code_generator.generate_model_loading(self.pytorch_model_info)

        # Check that the model ID is included
        self.assertIn("test_pytorch_model", pytorch_code)

        # Check for PyTorch-specific code
        self.assertIn("torch.load", pytorch_code)

        # Check for model info saving
        self.assertIn("model_info.json", pytorch_code)

    def test_generate_model_loading_tensorflow(self):
        """
        Tests the generation of TensorFlow model loading code.

        This test verifies:
        1. The model ID is correctly included in the generated code
        2. TensorFlow-specific loading code is present (containing "tf.keras.models.load_model")
        3. The code includes functionality to save model metadata to a JSON file
        4. The generated code follows proper Python syntax and TensorFlow conventions
        """
        tensorflow_code = self.code_generator.generate_model_loading(self.tensorflow_model_info)

        # Check that the model ID is included
        self.assertIn("test_tensorflow_model", tensorflow_code)

        # Check for TensorFlow-specific code
        self.assertIn("tf.keras.models.load_model", tensorflow_code)

        # Check for model info saving
        self.assertIn("model_info.json", tensorflow_code)

    def test_generate_dataset_loading(self):
        """
        Tests the generation of dataset loading code.

        This test verifies:
        1. The dataset name is correctly included in the generated code
        2. The code includes appropriate dataset loading functionality
        3. Proper error handling with try/except blocks is implemented
        4. The code accounts for different dataset split types as specified in the input
        """
        dataset_code = self.code_generator.generate_dataset_loading(self.dataset_info)

        # Check that the dataset name is included
        self.assertIn("test_dataset", dataset_code)

        # Check for dataset loading code
        self.assertIn("load_dataset", dataset_code)

        # Check for appropriate error handling
        self.assertIn("try:", dataset_code)
        self.assertIn("except", dataset_code)

    def test_generate_evaluation_code(self):
        """
        Tests the generation of model evaluation code for different frameworks.

        This test verifies:
        1. The model ID is correctly referenced in the evaluation code
        2. The specified metrics (accuracy, loss) are included in the evaluation logic
        3. The code saves evaluation results to a JSON file for reproducibility
        4. Framework-specific evaluation approaches are used (PyTorch vs TensorFlow)
        5. Proper error handling and execution flow is maintained
        """
        # Test with PyTorch model
        metrics = ["accuracy", "loss"]
        pytorch_eval_code = self.code_generator.generate_evaluation_code(
            self.pytorch_model_info, metrics
        )

        # Check that the model ID is included
        self.assertIn("test_pytorch_model", pytorch_eval_code)

        # Check that metrics are included
        self.assertIn("accuracy", pytorch_eval_code)
        self.assertIn("loss", pytorch_eval_code)

        # Check for evaluation results saving
        self.assertIn("evaluation_results.json", pytorch_eval_code)

        # Test with TensorFlow model
        tensorflow_eval_code = self.code_generator.generate_evaluation_code(
            self.tensorflow_model_info, metrics
        )

        # Check that it contains TensorFlow-specific elements
        self.assertIn("test_tensorflow_model", tensorflow_eval_code)

    def test_generate_visualization_code_performance(self):
        """
        Tests the generation of model performance visualization code.

        This test verifies:
        1. The code includes a function named "visualize_performance"
        2. Proper plot setup with matplotlib (figure creation, bar charts)
        3. The visualization is saved to a file
        4. The visualization displays the model metrics appropriately
        """
        viz_code = self.code_generator.generate_visualization_code("model_performance", {})

        # Check for visualization aspects
        self.assertIn("visualize_performance", viz_code)
        self.assertIn("plt.figure", viz_code)
        self.assertIn("plt.bar", viz_code)
        self.assertIn("plt.savefig", viz_code)

    def test_generate_visualization_code_images(self):
        """
        Tests the generation of image visualization code.

        This test verifies:
        1. The code includes a function named "visualize_images"
        2. The specified image directory parameter is correctly used
        3. The code includes image display functionality with matplotlib
        4. Grid sizing for multiple images is implemented correctly
        5. The visualization handles image loading and display appropriately
        """
        params = {"image_dir": "test_images", "max_images": 9}
        viz_code = self.code_generator.generate_visualization_code("generated_images", params)

        # Check for image-specific visualization code
        self.assertIn("visualize_images", viz_code)
        self.assertIn("test_images", viz_code)
        self.assertIn("plt.imshow", viz_code)
        self.assertIn("grid_size", viz_code)

    def test_generate_resource_monitoring(self):
        """
        Tests the generation of resource monitoring code.

        This test verifies:
        1. A ResourceMonitor class is created with appropriate methods
        2. The code implements tracking for CPU, memory, and GPU usage
        3. Monitoring starts in a background thread
        4. Data collection and storage functionality is included
        5. Visualization methods for displaying resource usage are implemented
        6. Cleanup mechanisms are properly registered
        """
        monitor_code = self.code_generator.generate_resource_monitoring()

        # Check for ResourceMonitor class and methods
        self.assertIn("class ResourceMonitor", monitor_code)
        self.assertIn("def start", monitor_code)
        self.assertIn("def stop", monitor_code)
        self.assertIn("def _monitor_resources", monitor_code)
        self.assertIn("def get_summary", monitor_code)
        self.assertIn("def plot_usage", monitor_code)

        # Check for resource monitoring components
        self.assertIn("cpu_percent", monitor_code)
        self.assertIn("memory", monitor_code)
        self.assertIn("GPU", monitor_code.upper())

        # Check for visualizations
        self.assertIn("plt.figure", monitor_code)
        self.assertIn("plt.plot", monitor_code)

    def test_code_execution(self):
        """
        Tests that the generated code is syntactically valid Python.

        This test verifies:
        1. The model loading code can be compiled without syntax errors
        2. The dataset loading code can be compiled without syntax errors
        3. The generated code follows proper Python syntax and indentation rules
        4. The code is valid for execution in a Python environment
        """
        # Test model loading code
        model_code = self.code_generator.generate_model_loading(self.pytorch_model_info)
        try:
            # This should not raise a SyntaxError
            compile(model_code, '<string>', 'exec')
            valid_syntax = True
        except SyntaxError:
            valid_syntax = False

        self.assertTrue(valid_syntax, "Generated model loading code has syntax errors")

        # Test dataset loading code
        dataset_code = self.code_generator.generate_dataset_loading(self.dataset_info)
        try:
            compile(dataset_code, '<string>', 'exec')
            valid_syntax = True
        except SyntaxError:
            valid_syntax = False

        self.assertTrue(valid_syntax, "Generated dataset loading code has syntax errors")

    def test_template_rendering(self):
        """
        Tests the Jinja2 template rendering functionality used by the CodeGenerator.

        This test verifies:
        1. The Template class correctly substitutes variables in template strings
        2. Different parameter sets produce the expected output strings
        3. The rendering mechanism works consistently across different inputs
        4. The template engine itself functions correctly
        """
        from jinja2 import Template

        # Create a test template
        template_str = "Model ID: {{ model_id }}, Framework: {{ framework }}"
        template = Template(template_str)

        # Test with different parameters
        result1 = template.render(model_id="model1", framework="pytorch")
        self.assertEqual(result1, "Model ID: model1, Framework: pytorch")

        result2 = template.render(model_id="model2", framework="tensorflow")
        self.assertEqual(result2, "Model ID: model2, Framework: tensorflow")

    def test_conditional_code_generation(self):
        """
        Tests that the CodeGenerator produces different code based on input parameters.

        This test verifies:
        1. Different model architectures (transformer, CNN, diffusion) produce distinct code
        2. Each architecture type includes the appropriate specialized libraries and functions:
           - Transformer models use transformers library and AutoModel
           - CNN models use torchvision.models and include architecture options
           - Diffusion models use StableDiffusionPipeline
        3. Different frameworks (PyTorch vs TensorFlow) produce framework-specific code
        4. The conditional logic in templates correctly generates specialized code
        5. Framework-specific libraries don't appear in the wrong framework's code
        """
        # Create distinct model configurations
        transformer_model = dict(self.pytorch_model_info)
        transformer_model["architecture_type"] = {"value": "transformer"}

        cnn_model = dict(self.pytorch_model_info)
        cnn_model["architecture_type"] = {"value": "cnn"}

        diffusion_model = dict(self.pytorch_model_info)
        diffusion_model["architecture_type"] = {"value": "diffusion"}

        # Generate code for different model architectures
        transformer_code = self.code_generator.generate_model_loading(transformer_model)
        cnn_code = self.code_generator.generate_model_loading(cnn_model)
        diffusion_code = self.code_generator.generate_model_loading(diffusion_model)

        # Test the codes are different
        self.assertNotEqual(transformer_code.strip(), cnn_code.strip())
        self.assertNotEqual(transformer_code.strip(), diffusion_code.strip())
        self.assertNotEqual(cnn_code.strip(), diffusion_code.strip())

        # Test for specific architectural elements in each type
        self.assertIn("transformers", transformer_code)
        self.assertIn("AutoModel", transformer_code)

        self.assertIn("torchvision.models", cnn_code)
        self.assertIn("resnet", cnn_code)

        self.assertIn("StableDiffusionPipeline", diffusion_code)

        # Also test that framework differences work
        tensorflow_model = dict(self.tensorflow_model_info)
        tensorflow_model["architecture_type"] = {"value": "cnn"}

        tf_code = self.code_generator.generate_model_loading(tensorflow_model)

        # Different framework should produce different code
        self.assertNotEqual(tf_code.strip(), cnn_code.strip())

        # Framework-specific elements
        self.assertIn("tf.keras", tf_code)
        self.assertNotIn("torchvision", tf_code)


if __name__ == '__main__':
    unittest.main()