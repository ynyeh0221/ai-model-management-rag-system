import re
from typing import Dict, List, Tuple, Optional

from graphviz import Digraph


class ModelDiagramGenerator:
    """Class for generating visual diagrams of neural network architectures."""

    def __init__(self, show_dimensions: bool = True, format: str = "png"):
        """
        Initialize the diagram generator.

        Args:
            show_dimensions: Whether to show dimension information in the diagram
            format: Output format (png, pdf, svg, etc.)
        """
        self.show_dimensions = show_dimensions
        self.format = format
        self.components = []
        self.component_layers = {}

    def _create_diagram(self) -> Digraph:
        """Create a Graphviz diagram of the model architecture."""
        if not self.components:
            print("ERROR: No components to visualize")
            # Create minimal diagram with error message
            dot = Digraph(comment='Model Architecture', format=self.format)
            dot.attr('graph', rankdir='TB')
            dot.node("empty", "No model architecture components detected",
                     shape="box", style="filled", color="red", fontcolor="white")
            return dot

        print(f"Creating diagram with {len(self.components)} components")

        dot = Digraph(comment='Model Architecture', format=self.format)
        dot.attr('graph', rankdir='TB', splines='ortho')  # Top to bottom layout
        dot.attr('node', shape='box', style='filled', color='lightblue',
                 fontname='Arial', fontsize='10', margin='0.1,0.1')
        dot.attr('edge', fontname='Arial', fontsize='8')

        # Keep track of nodes
        nodes = set()

        # Create component subgraphs
        for component in self.components:
            print(f"Processing component: {component}")
            try:
                with dot.subgraph(name=f'cluster_{component}') as c:
                    c.attr(label=component, style='filled', color='lightgrey', fontname='Arial', fontsize='12')

                    # Add layers as nodes within the component
                    for layer_name, layer_type, dimensions in self.component_layers[component]:
                        node_id = f"{component}_{layer_name}"

                        # Create label with or without dimensions
                        if self.show_dimensions and dimensions:
                            label = f"{layer_name}\n{layer_type}\n({dimensions})"
                        else:
                            label = f"{layer_name}\n{layer_type}"

                        print(f"  Adding node: {node_id} with label: {label}")
                        c.node(node_id, label=label)
                        nodes.add(node_id)

                        # If the layer_type matches a component name, add an edge
                        if layer_type.strip() in self.components:
                            print(f"  Adding component reference edge: {node_id} -> cluster_{layer_type.strip()}")
                            dot.edge(node_id, f"cluster_{layer_type.strip()}", style='dashed', color='red')
            except Exception as e:
                print(f"ERROR creating subgraph for {component}: {e}")

        # Add connections between layers within each component
        edge_count = 0
        for component in self.components:
            layers = self.component_layers[component]

            # Connect layers sequentially within the component
            for i in range(len(layers) - 1):
                current_layer = layers[i]
                next_layer = layers[i + 1]

                # Only connect layers that look like they should be connected
                if self._should_connect_layers(current_layer, next_layer):
                    src = f"{component}_{current_layer[0]}"
                    dst = f"{component}_{next_layer[0]}"
                    print(f"  Adding edge: {src} -> {dst}")
                    dot.edge(src, dst)
                    edge_count += 1

        print(f"Created diagram with {len(nodes)} nodes and approximately {edge_count} edges")

        # If no nodes were created, add a dummy node
        if not nodes:
            print("WARNING: No nodes were created, adding dummy node")
            dot.node("empty", "Empty Model Architecture", shape="box", style="filled", color="orange")

        return dot

    def _extract_architecture_section(self, ast_summary: str) -> str:
        """Extract the Model Architecture section from the AST summary."""
        match = re.search(r'Model Architecture:(.*?)$', ast_summary, re.DOTALL)
        if not match:
            return ""
        return "Model Architecture:" + match.group(1)

    def _parse_architecture(self, architecture_text: str) -> Tuple[List[str], Dict[str, List[Tuple[str, str, str]]]]:
        """
        Parse the architecture text into components and their layers.
        """
        components = []
        component_layers = {}

        current_component = None
        lines = architecture_text.split('\n')

        print(f"Parsing architecture with {len(lines)} lines")
        component_count = 0
        layer_count = 0

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith("Component: "):
                # Start of a new component
                current_component = line[len("Component: "):]
                components.append(current_component)
                component_layers[current_component] = []
                component_count += 1
                print(f"Found component: {current_component}")
            elif current_component is not None and ":" in line:
                # This matches lines like "  avg_pool: AdaptiveAvgPool2d(1)"
                # More flexible than the previous regex
                parts = line.split(":", 1)
                if len(parts) == 2:
                    layer_name = parts[0].strip()
                    layer_info = parts[1].strip()

                    # Extract layer type and dimensions
                    type_dim_match = re.match(r'([^\(]+)(?:\(([^\)]*)\))?', layer_info)
                    if type_dim_match:
                        layer_type = type_dim_match.group(1).strip()
                        dimensions = type_dim_match.group(2) if type_dim_match.group(2) else ""

                        component_layers[current_component].append((layer_name, layer_type, dimensions))
                        layer_count += 1
                        print(f"  Extracted layer: {layer_name}: {layer_type}({dimensions})")

        print(f"Parsing complete: {component_count} components, {layer_count} layers")
        return components, component_layers

    def _should_connect_layers(self, layer1: Tuple[str, str, str], layer2: Tuple[str, str, str]) -> bool:
        """
        Determine if two layers should be connected in the diagram.
        This is a heuristic and may need refinement based on specific models.
        """
        # Sequential connectivity patterns - typical patterns for different architectures
        sequential_patterns = [
            # Encoder/Decoder patterns
            {'pattern': ["initial_conv", "down1", "down2", "down3", "down4", "fc_mu", "fc_logvar"]},
            {'pattern': ["fc", "up4", "up3", "up2", "up1", "final_conv"]},

            # Conv/Linear patterns
            {'pattern': ["conv1", "conv2", "conv3", "fc1", "fc2"]},

            # UNet patterns
            {'pattern': ["enc1", "enc2", "bottleneck", "dec1", "dec2"]},

            # Transformer patterns
            {'pattern': ["embedding", "encoder", "decoder", "output"]}
        ]

        name1, type1, _ = layer1
        name2, type2, _ = layer2

        # Check for layers that have sequential naming patterns
        for pattern_set in sequential_patterns:
            pattern = pattern_set['pattern']
            if name1 in pattern and name2 in pattern:
                idx1 = pattern.index(name1)
                idx2 = pattern.index(name2)
                if idx2 == idx1 + 1:  # Direct sequential connection
                    return True

        # Connect layers whose names suggest sequential order
        # (e.g., layer1 -> layer2, conv1 -> conv2)
        base_name1 = re.match(r'([a-zA-Z_]+)(\d+)', name1)
        base_name2 = re.match(r'([a-zA-Z_]+)(\d+)', name2)

        if base_name1 and base_name2 and base_name1.group(1) == base_name2.group(1):
            num1 = int(base_name1.group(2))
            num2 = int(base_name2.group(2))
            if num2 == num1 + 1:
                return True

        # Check for special pairs that should be connected
        special_pairs = [
            ("encoder", "decoder"),
            ("time_embedding", "class_embedding"),
            ("project_in", "enc1"),
            ("bottleneck", "dec1"),
            ("dec2", "project_out")
        ]

        if (name1, name2) in special_pairs:
            return True

        return False

    def generate_diagram(self, ast_summary: str, output_file: str = "model_architecture") -> Optional[str]:
        """
        Generate a diagram from an AST summary.

        Args:
            ast_summary: The output from ASTSummaryGenerator.generate_summary()
            output_file: Base name for the output file (without extension)

        Returns:
            Path to the generated diagram file or None if generation failed
        """
        # Extract and parse the model architecture
        architecture_section = self._extract_architecture_section(ast_summary)
        print(f"Extracted architecture section length: {len(architecture_section)}")
        print(f"First 200 chars: {architecture_section[:200]}...")

        if not architecture_section:
            print("ERROR: No architecture section found in AST summary")
            return None

        # Parse the architecture into components and layers
        self.components, self.component_layers = self._parse_architecture(architecture_section)
        print(
            f"Found {len(self.components)} components with {sum(len(layers) for layers in self.component_layers.values())} total layers")

        # Debug component info
        for comp in self.components:
            print(f"Component: {comp} - {len(self.component_layers.get(comp, []))} layers")
            for layer in self.component_layers.get(comp, [])[:3]:  # Show first 3 layers
                print(f"  - {layer[0]}: {layer[1]}({layer[2]})")

        # Create and render the diagram
        dot = self._create_diagram()
        if dot:
            try:
                rendered_file = dot.render(output_file, format=self.format, cleanup=True)
                print(f"Diagram rendered to: {rendered_file}")
                return rendered_file
            except Exception as e:
                print(f"ERROR rendering diagram: {e}")
                return None
        else:
            print("ERROR: Failed to create diagram object")
            return None


# Helper function for easy use
def draw_model_architecture(ast_summary: str, output_path: str = "model_diagram.png",
                            show_dimensions: bool = True) -> str:
    """
    User-friendly wrapper to generate and save a model architecture diagram.

    Args:
        ast_summary: The AST summary text containing the model architecture
        output_path: Where to save the diagram (should end with .png, .pdf, or .svg)
        show_dimensions: Whether to show dimension information in the diagram

    Returns:
        Path to the saved diagram or error message
    """
    # Determine format from output_path
    format = output_path.split('.')[-1]
    if format not in ['png', 'pdf', 'svg']:
        format = 'png'  # Default to PNG

    output_file = output_path.rsplit('.', 1)[0]

    # Generate the diagram
    generator = ModelDiagramGenerator(show_dimensions=show_dimensions, format=format)
    result = generator.generate_diagram(ast_summary, output_file)

    if result:
        return f"Model architecture diagram saved to {output_path}"
    else:
        return "No model architecture found in the AST summary"