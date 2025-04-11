import datetime
import os
import re

import jinja2


class TemplateManager:
    def __init__(self, templates_dir="./templates"):
        self.templates_dir = templates_dir
        self.templates = {}  # { template_id: { version: content, ... }, ... }
        self.history = {}    # { template_id: [ { "version": version, "timestamp": timestamp }, ... ], ... }
        self._load_templates()
    
    def _load_templates(self):
        """Load templates from the templates directory."""
        # Create the templates directory if it doesn't exist.
        if not os.path.exists(self.templates_dir):
            os.makedirs(self.templates_dir)
        
        # Pattern expects filenames like "template_id_v1.0.tpl"
        pattern = re.compile(r'^(?P<template_id>.+)_v(?P<version>[\d\.]+)\.tpl$')
        for filename in os.listdir(self.templates_dir):
            file_path = os.path.join(self.templates_dir, filename)
            if os.path.isfile(file_path):
                match = pattern.match(filename)
                if match:
                    template_id = match.group("template_id")
                    version = match.group("version")
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    if template_id not in self.templates:
                        self.templates[template_id] = {}
                        self.history[template_id] = []
                    
                    self.templates[template_id][version] = content
                    # Record history using file's last modification time.
                    mod_time = os.path.getmtime(file_path)
                    self.history[template_id].append({
                        "version": version,
                        "timestamp": datetime.datetime.fromtimestamp(mod_time).isoformat()
                    })
    
    def _version_to_tuple(self, version):
        """Convert a version string like '1.0' to a tuple of integers (1, 0)."""
        try:
            return tuple(map(int, version.split('.')))
        except Exception:
            return (0,)
    
    def _get_latest_version(self, template_id):
        """Return the latest version available for a template_id."""
        versions = list(self.templates.get(template_id, {}).keys())
        if not versions:
            return None
        # Sort versions by converting them to tuples.
        versions.sort(key=self._version_to_tuple)
        return versions[-1]
    
    def get_template(self, template_id, version=None):
        """Get a template by ID and optionally version."""
        if template_id not in self.templates:
            return None
        if version:
            return self.templates[template_id].get(version)
        else:
            latest_version = self._get_latest_version(template_id)
            return self.templates[template_id].get(latest_version)
    
    def render_template(self, template_id, context, version=None):
        """Render a template with the given context."""
        template_content = self.get_template(template_id, version)
        if template_content is None:
            raise ValueError(f"Template '{template_id}' with version '{version}' not found.")
        jinja_template = jinja2.Template(template_content)
        return jinja_template.render(context)
    
    def save_template(self, template_id, content, version=None):
        """Save a template."""
        # If no version provided, determine next version.
        if template_id in self.templates:
            if version is None:
                latest_version = self._get_latest_version(template_id)
                if latest_version:
                    # Simple version increment: e.g., "1.0" becomes "1.1".
                    latest_parts = self._version_to_tuple(latest_version)
                    if len(latest_parts) > 1:
                        new_version = f"{latest_parts[0]}.{latest_parts[1] + 1}"
                    else:
                        new_version = f"{latest_parts[0] + 1}.0"
                else:
                    new_version = "1.0"
            else:
                new_version = version
        else:
            self.templates[template_id] = {}
            self.history[template_id] = []
            new_version = version if version is not None else "1.0"
        
        filename = f"{template_id}_v{new_version}.tpl"
        file_path = os.path.join(self.templates_dir, filename)
        with open(file_path, 'w') as f:
            f.write(content)
        
        # Update in-memory store and history.
        self.templates[template_id][new_version] = content
        timestamp = datetime.datetime.now().isoformat()
        self.history[template_id].append({
            "version": new_version,
            "timestamp": timestamp
        })
        return new_version
    
    def get_template_history(self, template_id):
        """Get the history of a template."""
        return self.history.get(template_id, [])

