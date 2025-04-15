import datetime
import os
import sys
import tempfile
import textwrap
import unittest

# Ensure import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.document_processor.code_parser import CodeParser


class TestCodeParser(unittest.TestCase):

    def setUp(self):
        self.parser = CodeParser()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_file_path = os.path.join(self.temp_dir.name, "test_model.py")

        sample_content = """
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.num_layers = 3
        self.hidden_size = 256
        self.num_attention_heads = 8

        self.layers = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.layers(x)

dataset = "CIFAR-10"
train_data = {"num_samples": 50000, "split": "train"}

batch_size = 64
learning_rate = 0.001
optimizer = "Adam"
epochs = 100

accuracy = 0.92
loss = 0.08
perplexity = 1.5
eval_dataset = "CIFAR-10-test"
"""
        with open(self.test_file_path, 'w') as f:
            f.write(sample_content)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_parse_extension_filtering(self):
        result = self.parser.parse(self.test_file_path)
        self.assertIsNotNone(result)

        non_py_file = os.path.join(self.temp_dir.name, "not_python.txt")
        with open(non_py_file, 'w') as f:
            f.write("This is not Python code")

        result = self.parser.parse(non_py_file)
        self.assertIsNone(result)

    def test_parse_file_basic_metadata(self):
        model_info = self.parser.parse_file(self.test_file_path)
        self.assertIn("creation_date", model_info)
        self.assertIn("last_modified_date", model_info)
        self.assertEqual(model_info["model_id"], "unknown")
        self.assertEqual(model_info["model_family"], "unknown")
        self.assertEqual(model_info["version"], "unknown")
        self.assertTrue(model_info["is_model_script"])

    def test_framework_detection(self):
        model_info = self.parser.parse_file(self.test_file_path)
        framework = model_info.get("framework", {})
        self.assertIsInstance(framework, dict)
        self.assertIn("name", framework)
        self.assertIn("version", framework)
        self.assertIsInstance(framework["name"], str)
        self.assertIsInstance(framework["version"], str)

    def test_architecture_extraction(self):
        model_info = self.parser.parse_file(self.test_file_path)
        architecture = model_info.get("architecture", {})
        self.assertIsInstance(architecture, dict)
        self.assertIn("dimensions", architecture)
        self.assertIsInstance(architecture["dimensions"], dict)

    def test_dataset_extraction(self):
        model_info = self.parser.parse_file(self.test_file_path)
        dataset = model_info.get("dataset", {})
        self.assertIsInstance(dataset, dict)
        self.assertIn("name", dataset)
        self.assertIsInstance(dataset["name"], str)
        if "num_samples" in dataset:
            self.assertIsInstance(dataset["num_samples"], int)
        if "split" in dataset:
            self.assertIsInstance(dataset["split"], str)

    def test_training_config_extraction(self):
        model_info = self.parser.parse_file(self.test_file_path)
        config = model_info.get("training_config", {})
        self.assertIsInstance(config, dict)

        # Validate keys if they exist
        for key in ["batch_size", "learning_rate", "optimizer", "epochs"]:
            if key in config:
                self.assertIsInstance(config[key], (int, float, str))

    def test_performance_metrics_extraction(self):
        model_info = self.parser.parse_file(self.test_file_path)
        performance = model_info.get("performance", {})
        self.assertIsInstance(performance, dict)

        for key in ["accuracy", "loss", "perplexity", "eval_dataset"]:
            if key in performance:
                if key == "eval_dataset":
                    self.assertIsInstance(performance[key], str)
                else:
                    self.assertIsInstance(performance[key], float)

    def test_split_ast_and_subsplit_chunks(self):
        structured_code = textwrap.dedent("""
            import torch

            class MyModel(torch.nn.Module):
                def __init__(self):
                    super(MyModel, self).__init__()
                    self.fc1 = torch.nn.Linear(100, 256)
                    self.fc2 = torch.nn.Linear(256, 10)
                    self.dropout = torch.nn.Dropout(0.1)
                    self.activation = torch.nn.ReLU()

                def forward(self, x):
                    x = self.activation(self.fc1(x))
                    x = self.dropout(x)
                    return self.fc2(x)

            def train(model, data_loader, optimizer):
                model.train()
                for inputs, targets in data_loader:
                    optimizer.zero_grad()
                    output = model(inputs)
                    loss = torch.nn.functional.cross_entropy(output, targets)
                    loss.backward()
                    optimizer.step()

            def evaluate(model, val_loader):
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        output = model(inputs)
                        preds = torch.argmax(output, dim=1)
                        correct += (preds == targets).sum().item()
                        total += targets.size(0)
                return correct / total

            learning_rate = 0.001
            batch_size = 64
        """)

        chunks = self.parser.split_ast_and_subsplit_chunks(structured_code, chunk_size=250, overlap=50)
        self.assertGreaterEqual(len(chunks), 3)
        for chunk in chunks:
            self.assertIn("text", chunk)
            self.assertIn("source_block", chunk)
            self.assertIn("offset", chunk)
            self.assertIsInstance(chunk["text"], str)
            self.assertTrue(len(chunk["text"].strip()) > 0)

        joined_code = "\n".join(chunk["text"] for chunk in chunks)
        self.assertIn("class MyModel", joined_code)
        self.assertIn("def train", joined_code)
        self.assertIn("def evaluate", joined_code)
        self.assertIn("learning_rate", joined_code)

    def test_git_date_extraction(self):
        date = self.parser._get_creation_date(self.test_file_path)
        self.assertIsNotNone(date)
        try:
            datetime.datetime.fromisoformat(date)
        except ValueError:
            self.fail("Date is not in ISO format")

    def test_syntax_error_handling(self):
        syntax_error_file = os.path.join(self.temp_dir.name, "syntax_error.py")
        with open(syntax_error_file, 'w') as f:
            f.write("This is not valid Python syntax :")

        with self.assertRaises(ValueError):
            self.parser.parse_file(syntax_error_file)


if __name__ == '__main__':
    unittest.main()
