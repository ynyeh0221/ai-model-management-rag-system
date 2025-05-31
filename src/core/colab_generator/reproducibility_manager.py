import hashlib
import os
import platform
import socket
import tempfile
from datetime import datetime, timezone

import nbformat
from nbconvert import HTMLExporter, PDFExporter
from traitlets.config import Config


class ReproducibilityManager:
    def __init__(self):
        # __init__ is deliberately empty because there is no state to initialize.
        # If we later need to cache or configure something, we can add it here.
        pass

    def generate_execution_log(self, notebook, parameters):
        """
        Generate a metadata-rich execution log for a notebook.

        Args:
            notebook (nbformat.NotebookNode): The executed notebook object.
            parameters (dict): Execution parameters used in this run.

        Returns:
            dict: Execution log containing hash, timestamp (UTC), parameters, etc.
        """
        # Compute a SHA256 hash of the notebook contents
        digest = self.calculate_hash_digest(notebook)

        # Use a timezone-aware UTC timestamp instead of datetime.utcnow()
        now_utc = datetime.now(timezone.utc)
        timestamp = now_utc.isoformat()

        log = {
            "execution_id": f"exec_{digest[:8]}_{int(now_utc.timestamp())}",
            "timestamp": timestamp,
            "notebook_hash": digest,
            "parameters": parameters,
            "machine": {
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "python_version": platform.python_version(),
            },
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
        # Serialize the notebook to a string, then hash it
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

        # Again, use timezone-aware UTC timestamp
        now_utc = datetime.now(timezone.utc)

        env_snapshot = {
            "execution_id": execution_id,
            "timestamp": now_utc.isoformat(),
            "system_info": {
                "os": platform.system(),
                "os_version": platform.version(),
                "python_version": platform.python_version(),
                "hostname": socket.gethostname(),
            },
            "packages": packages,
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
            output_path = os.path.join(
                tempfile.gettempdir(), "notebook_export.html"
            )

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
            output_path = os.path.join(
                tempfile.gettempdir(), "notebook_export.pdf"
            )

        with open(output_path, "wb") as f:
            f.write(pdf_data)

        return output_path

    def add_reproducibility_info(self, notebook, model_id):
        """
        Embed reproducibility metadata into the notebook's metadata field.

        Args:
            notebook (nbformat.NotebookNode): The notebook object.
            model_id (str): ID of the model for traceability.

        Returns:
            nbformat.NotebookNode: Updated notebook with reproducibility info.
        """
        log = self.generate_execution_log(
            notebook, parameters={"model_id": model_id}
        )
        notebook.metadata["reproducibility"] = {
            "execution_log": log,
            "environment": self.record_environment(log["execution_id"]),
        }
        return notebook
