import os
import hashlib
import nbformat
import platform
import socket
import tempfile
from datetime import datetime
from nbconvert import HTMLExporter, PDFExporter
from traitlets.config import Config


class ReproducibilityManager:
    def __init__(self):
        pass

    def generate_execution_log(self, notebook, parameters):
        """
        Generate a metadata-rich execution log for a notebook.

        Args:
            notebook (nbformat.NotebookNode): The executed notebook object.
            parameters (dict): Execution parameters used in this run.

        Returns:
            dict: Execution log containing hash, timestamp, parameters, etc.
        """
        digest = self.calculate_hash_digest(notebook)
        timestamp = datetime.utcnow().isoformat()
        log = {
            "execution_id": f"exec_{digest[:8]}_{int(datetime.utcnow().timestamp())}",
            "timestamp": timestamp,
            "notebook_hash": digest,
            "parameters": parameters,
            "machine": {
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "python_version": platform.python_version()
            }
        }
        return log

    def calculate_hash_digest(self, notebook):
        """
        Calculate a SHA256 hash digest of the notebook contents.

        Args:
            notebook (nbformat.NotebookNode): The notebook object.

        Returns:
            str: SHA256 hex digest.
        """
        nb_str = nbformat.writes(notebook)
        return hashlib.sha256(nb_str.encode("utf-8")).hexdigest()

    def record_environment(self, execution_id):
        """
        Record environment details for the execution (installed packages, system info).

        Args:
            execution_id (str): Unique ID for the execution.

        Returns:
            dict: Snapshot of the Python environment.
        """
        try:
            import pkg_resources
            packages = sorted(
                [f"{d.project_name}=={d.version}" for d in pkg_resources.working_set]
            )
        except ImportError:
            packages = ["Could not retrieve installed packages"]

        env_snapshot = {
            "execution_id": execution_id,
            "timestamp": datetime.utcnow().isoformat(),
            "system_info": {
                "os": platform.system(),
                "os_version": platform.version(),
                "python_version": platform.python_version(),
                "hostname": socket.gethostname()
            },
            "packages": packages
        }
        return env_snapshot

    def export_to_html(self, notebook, output_path=None):
        """
        Export a notebook to an HTML file.

        Args:
            notebook (nbformat.NotebookNode): The notebook to export.
            output_path (str): Optional path to save the HTML.

        Returns:
            str: Path to the saved HTML file.
        """
        exporter = HTMLExporter()
        body, _ = exporter.from_notebook_node(notebook)

        if not output_path:
            output_path = os.path.join(tempfile.gettempdir(), "notebook_export.html")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(body)

        return output_path

    def export_to_pdf(self, notebook, output_path=None):
        """
        Export a notebook to a PDF file.

        Args:
            notebook (nbformat.NotebookNode): The notebook to export.
            output_path (str): Optional path to save the PDF.

        Returns:
            str: Path to the saved PDF file.
        """
        # You must have TeX installed for PDF conversion to work!
        c = Config()
        c.PDFExporter.latex_count = False
        exporter = PDFExporter(config=c)

        pdf_data, _ = exporter.from_notebook_node(notebook)

        if not output_path:
            output_path = os.path.join(tempfile.gettempdir(), "notebook_export.pdf")

        with open(output_path, "wb") as f:
            f.write(pdf_data)

        return output_path
