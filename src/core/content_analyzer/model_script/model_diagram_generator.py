import re
from typing import Dict, List, Tuple, Optional, Set
from graphviz import Digraph


class ModelDiagramGenerator:
    """Generates visual diagrams for neural network architectures."""

    def __init__(
        self,
        show_dimensions: bool = True,
        output_format: str = "png"
    ):
        self.show_dimensions = show_dimensions
        self.format = output_format
        self.components: List[str] = []
        self.component_layers: Dict[str, List[Tuple[str, str, str]]] = {}
        self.dependencies: Dict[str, Set[str]] = {}
        self.root_components: List[str] = []

    def generate_diagram(
        self,
        ast_summary: str,
        output_file: str = "model_architecture"
    ) -> Optional[str]:
        """
        Parse an AST summary and render the model diagram.
        Returns path to the rendered file or None on failure.
        """
        # Extract the Model Architecture section
        arch_section = self._extract_architecture_section(ast_summary)
        print(f"Extracted architecture section length: {len(arch_section)}")
        if not arch_section:
            print("ERROR: No Model Architecture section found in AST summary")
            return None

        # Parse into components and layers
        self.components, self.component_layers = self._parse_architecture(arch_section)
        if not self.components:
            print("ERROR: No components detected in architecture.")
            return None

        # Infer dependencies and roots if not explicitly provided
        if not self.dependencies:
            self._infer_dependencies()
        if not self.root_components:
            self._infer_root_components()

        print(
            f"Found {len(self.components)} components with "
            f"{sum(len(l) for l in self.component_layers.values())} total layers"
        )

        # Build and render diagram
        dot = self._build_diagram()
        try:
            rendered_path = dot.render(output_file, format=self.format, cleanup=True)
            print(f"Diagram rendered to: {rendered_path}")
            return rendered_path
        except Exception as e:
            print(f"ERROR rendering diagram: {e}")
            return None

    def _extract_architecture_section(self, text: str) -> str:
        """Extract the Model Architecture section from summary."""
        match = re.search(r'Model Architecture:(.*?)$', text, re.DOTALL)
        return match.group(0).strip() if match else ""

    def _parse_architecture(
        self,
        section: str
    ) -> Tuple[List[str], Dict[str, List[Tuple[str, str, str]]]]:
        """
        Convert raw architecture text into components and layer lists.
        """
        components: List[str] = []
        layers_map: Dict[str, List[Tuple[str, str, str]]] = {}
        current_comp: Optional[str] = None

        lines = section.splitlines()
        print(f"Parsing architecture with {len(lines)} lines")
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("Component:"):
                current_comp = stripped.split("Component:", 1)[1].strip()
                components.append(current_comp)
                layers_map[current_comp] = []
                print(f"Found component: {current_comp}")
            elif current_comp and ":" in stripped:
                name, info = map(str.strip, stripped.split(':', 1))
                m = re.match(r'([^\(]+)(?:\(([^)]*)\))?', info)
                if m:
                    layer_type = m.group(1).strip()
                    dims = m.group(2) or ""
                    layers_map[current_comp].append((name, layer_type, dims))
                    print(f"  Extracted layer: {name}: {layer_type}({dims})")

        print(
            f"Parsing complete: {len(components)} components, "
            f"{sum(len(v) for v in layers_map.values())} layers"
        )
        return components, layers_map

    def _infer_dependencies(self):
        """Link components based on layer-type references."""
        deps: Dict[str, Set[str]] = {c: set() for c in self.components}
        for comp, layers in self.component_layers.items():
            for _, layer_type, _ in layers:
                if layer_type in self.components:
                    deps[comp].add(layer_type)
        self.dependencies = deps
        if not any(deps.values()):
            print("NOTE: No component dependencies inferred.")

    def _infer_root_components(self):
        """Identify components not used by any other."""
        used = {d for deps in self.dependencies.values() for d in deps}
        self.root_components = [c for c in self.components if c not in used]
        if not self.root_components:
            print("NOTE: Could not identify root components.")

    def _build_diagram(self) -> Digraph:
        """Assemble the Graphviz Digraph."""
        dot = Digraph(comment='Model Architecture', format=self.format)
        dot.attr('graph', rankdir='TB', splines='ortho')
        dot.attr('node', shape='box', style='filled', color='lightblue', fontname='Arial', fontsize='10', margin='0.1,0.1')
        dot.attr('edge', fontname='Arial', fontsize='8')

        # Create component subgraphs
        nodes: Set[str] = set()
        for comp in self.components:
            with dot.subgraph(name=f'cluster_{comp}') as c:
                c.attr(label=comp, style='filled', color='lightgrey', fontname='Arial', fontsize='12')
                if comp in self.root_components:
                    c.attr(color='darkgreen', penwidth='2.0')
                for name, ltype, dims in self.component_layers.get(comp, []):
                    node_id = f"{comp}_{name}"
                    label = f"{name}\n{ltype}" + (f"\n({dims})" if self.show_dimensions and dims else "")
                    c.node(node_id, label=label)
                    nodes.add(node_id)

        # Dependency edges
        if self.dependencies:
            for comp, deps in self.dependencies.items():
                for dep in deps:
                    if dep in self.components:
                        # Only connect if both components have layers
                        if self.component_layers.get(comp) and self.component_layers.get(dep):
                            src = f"{comp}_{self.component_layers[comp][0][0]}"
                            dst = f"{dep}_{self.component_layers[dep][0][0]}"
                            dot.edge(src, dst, style='dashed', color='blue', label='uses')
                        else:
                            print(f"Skipping dependency edge {comp}->{dep}: missing layers in source or target")
        else:
            print("Skipping dependency edges: no dependencies defined.")

        # Layer-type reference edges
        for comp, layers in self.component_layers.items():
            for name, ltype, _ in layers:
                if ltype in self.components:
                    if self.component_layers.get(ltype):
                        src = f"{comp}_{name}"
                        dst = f"{ltype}_{self.component_layers[ltype][0][0]}"
                        dot.edge(src, dst, style='dashed', color='red', label='instance')
                    else:
                        print(f"Skipping reference edge {comp}_{name}->{ltype}: target missing layers")

        if not nodes:
            print("WARNING: No nodes created, adding dummy node")
            dot.node("empty", "Empty Model Architecture", shape="box", style="filled", color="orange")

        return dot


def draw_model_architecture(
    ast_summary: str,
    output_path: str = "model_diagram.png",
    show_dimensions: bool = True,
    component_tree: Optional[Dict[str, List[str]]] = None,
    root_component: Optional[str] = None
) -> str:
    """Wrapper: generates and saves diagram with optional explicit dependencies/roots."""
    fmt = output_path.rsplit('.', 1)[-1]
    generator = ModelDiagramGenerator(show_dimensions=show_dimensions, output_format=fmt)

    # Apply provided component tree
    if component_tree is not None:
        deps: Dict[str, Set[str]] = {}
        for comp, deps_list in component_tree.items():
            deps[comp] = set(d for d in deps_list if d in component_tree)
        generator.dependencies = deps

    # Apply provided root component(s)
    if root_component is not None:
        generator.root_components = [root_component] if isinstance(root_component, str) else list(root_component)

    result = generator.generate_diagram(ast_summary, output_file=output_path.rsplit('.', 1)[0])
    return f"Model architecture diagram saved to {output_path}" if result else "No model architecture found in the AST summary"