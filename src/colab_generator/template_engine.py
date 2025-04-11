import os

import nbformat
from jinja2 import Environment, FileSystemLoader, TemplateNotFound


class NotebookTemplateEngine:
    def __init__(self, templates_dir="./notebook_templates"):
        self.templates_dir = templates_dir
        self._load_templates()

    def _load_templates(self):
        """Initialize the Jinja2 environment."""
        if not os.path.exists(self.templates_dir):
            raise FileNotFoundError(f"Template directory not found: {self.templates_dir}")

        self.env = Environment(
            loader=FileSystemLoader(self.templates_dir),
            autoescape=False,  # not needed for code
            trim_blocks=True,
            lstrip_blocks=True
        )

    def list_templates(self):
        """List all available Jinja2 notebook templates (.ipynb.j2)."""
        templates = [
            f for f in os.listdir(self.templates_dir)
            if f.endswith(".ipynb.j2")
        ]
        return [os.path.splitext(os.path.splitext(t)[0])[0] for t in templates]

    def get_template(self, template_id):
        """Get a raw Jinja2 template object by ID (without .ipynb.j2)."""
        try:
            return self.env.get_template(f"{template_id}.ipynb.j2")
        except TemplateNotFound:
            raise FileNotFoundError(f"Template '{template_id}' not found in {self.templates_dir}")

    def render_template(self, template_id, context):
        template = self.get_template(template_id)
        rendered_str = template.render(**context)

        # TEMP DEBUG: save the raw rendered content to a file
        # with open("./notebooks/debug_rendered_notebook.ipynb", "w", encoding="utf-8") as f:
        #     f.write(rendered_str)
        # print("[DEBUG] Saved raw rendered notebook to debug_rendered_notebook.ipynb")

        try:
            notebook = nbformat.reads(rendered_str, as_version=4)
            return notebook
        except Exception as e:
            raise ValueError(
                f"Failed to parse rendered notebook: {e}\n\n"
                "Check debug_rendered_notebook.ipynb for raw output."
            )
