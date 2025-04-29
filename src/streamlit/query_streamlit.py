import asyncio
import logging
import re
import time

import streamlit as st
from src.cli.cli_response_utils.image_display_manager import ImageDisplayManager
from src.cli.cli_response_utils.llm_response_processor import LLMResponseProcessor
from src.cli.cli_response_utils.model_display_manager import ModelDisplayManager
from src.core.notebook_generator import NotebookGenerator
from src.core.rag_system import RAGSystem

logging.basicConfig(level=logging.INFO)


def set_streamlit_page_config():
    st.set_page_config(page_title="RAG System", page_icon="🔍", layout="wide")


class StreamlitInterface:
    def __init__(self):
        self.rag = RAGSystem()
        if 'logged_in' not in st.session_state:
            st.session_state.update({
                'logged_in': False,
                'status': 'idle',
                'logs': [],
                'results': None,
                'user_id': 'anonymous'
            })

        # Initialize display managers and processors (like in CLIInterface)
        self.model_display_manager = None
        self.image_display_manager = None
        self.notebook_generator = None
        self.llm_response_processor = None

    def login(self, comps):
        uid = st.session_state.user_id
        ok = self.rag.initialize(comps, uid)
        if ok:
            # make sure core sees the components
            self.rag.components = comps
            self._reg_cb()
            self._initialize_components()  # Initialize UI components
            st.session_state.logged_in = True
        return ok

    def _initialize_components(self):
        """Initialize component classes needed by the UI"""
        # Create display managers
        self.model_display_manager = ModelDisplayManager()
        self.image_display_manager = ImageDisplayManager()

        # Create generators and processors
        self.notebook_generator = NotebookGenerator()
        self.llm_response_processor = LLMResponseProcessor()

    def _reg_cb(self):
        self.rag.register_callback("on_log", lambda d: st.session_state.logs.append((time.time(), d)))
        self.rag.register_callback("on_status", lambda s: st.session_state.__setitem__('status', s))
        self.rag.register_callback("on_result", lambda r: st.session_state.__setitem__('results', r))
        self.rag.register_callback("on_error", lambda e: st.error(e))

    async def process_query(self, q):
        if not st.session_state.logged_in:
            st.error("Not logged in")
            return
        await self.rag.process_query(q,
                                     enable_clarity_check=st.session_state.get('enable_clarity_check', False),
                                     enable_comparison_detection=st.session_state.get('enable_comparison_detection',
                                                                                      True)
                                     )

    def render(self):
        """
        Display results in a streamlit-appropriate way
        Using proper Streamlit UI components instead of text-based tables
        """
        r = st.session_state.results
        if not r: return
        t = r.get("type", "")

        if t == "text_search":
            st.header("Answer")

            # Check if the response contains thinking tags (either <thinking> or <think>)
            response_text = r["final_response"]
            thinking_pattern = re.compile(r'<(thinking|think)>(.*?)</\1>', re.DOTALL)
            thinking_matches = thinking_pattern.findall(response_text)

            if thinking_matches:
                # If thinking tags are found, separate thinking from the response
                for tag_type, thinking_text in thinking_matches:
                    # Display thinking text with dark blue color on the same light blue background
                    st.markdown(f"""<div style='background-color:#e6f3ff; padding:10px; border-radius:5px; 
                                    margin-bottom:10px; color:#003366;'><strong>Thinking:</strong><br>{thinking_text}</div>""",
                                unsafe_allow_html=True)

                # Remove thinking tags from the main response
                clean_response = thinking_pattern.sub('', response_text)
                st.write(clean_response)
            else:
                # If no thinking tags, display the response as is
                st.write(response_text)

            # Display search results in a proper Streamlit table
            if "search_results" in r and len(r["search_results"]) > 0:
                st.subheader("Search Results")

                # Prepare data for Streamlit DataFrame
                search_data = []
                for i, result in enumerate(r["search_results"]):
                    metadata = result.get('metadata', {})
                    # Get file metadata
                    if isinstance(metadata.get('file'), str):
                        try:
                            import json
                            file_data = json.loads(metadata.get('file', '{}'))
                        except:
                            file_data = {}
                    else:
                        file_data = metadata.get('file', {})

                    # Extract data points
                    search_data.append({
                        "Rank": i + 1,
                        "Model ID": result.get('model_id', result.get('id', f'Item {i + 1}')),
                        "Score": result.get('score', result.get('similarity', result.get('rank_score', 'N/A'))),
                        "Created At": result.get('created_at', result.get('created_at', 'N/A')),
                        "Last Modified At": result.get('last_modified_at', result.get('last_modified_at', 'N/A')),
                        "Path": file_data.get('absolute_path', metadata.get('absolute_path', 'N/A')),
                        "Description": result.get('merged_description', 'N/A')[:100] + (
                            '...' if len(result.get('merged_description', '')) > 100 else '')
                    })

                # Create and display the DataFrame
                import pandas as pd
                df = pd.DataFrame(search_data)
                st.dataframe(df, use_container_width=True)

                # Show detailed information for the top result
                if len(r["search_results"]) > 0:
                    with st.expander("Top Result Details"):
                        top_result = r["search_results"][0]
                        metadata = top_result.get('metadata', {})

                        # Parse nested JSON if needed
                        for field in ["file", "framework", "architecture", "dataset", "training_config"]:
                            if isinstance(metadata.get(field), str):
                                try:
                                    import json
                                    parsed = json.loads(metadata.get(field, '{}'))
                                    metadata[field] = parsed
                                except:
                                    pass

                        # Display file info
                        if metadata.get('file'):
                            st.subheader("File Information")
                            file_info = metadata.get('file', {})
                            file_cols = st.columns(3)
                            with file_cols[0]:
                                st.metric("Size", f"{int(file_info.get('size_bytes', 0) / 1024)} KB")
                            with file_cols[1]:
                                st.metric("Created", file_info.get('creation_date', 'N/A')[:10] if isinstance(
                                    file_info.get('creation_date'), str) else 'N/A')
                            with file_cols[2]:
                                st.metric("Modified", file_info.get('last_modified_date', 'N/A')[:10] if isinstance(
                                    file_info.get('last_modified_date'), str) else 'N/A')

                        # Display model component diagram if available
                        if metadata.get('diagram_path'):
                            st.subheader("Model Component Diagram")
                            diagram_path = metadata.get('diagram_path')

                            # Handle diagram path appropriately based on its type
                            try:
                                import os
                                import json
                                from PIL import Image
                                import base64

                                # Variable to store the actual path for both display and full-size view
                                actual_image_path = None

                                # Resolve the actual image path from various possible formats
                                if isinstance(diagram_path, str) and diagram_path.startswith(
                                        '{') and diagram_path.endswith('}'):
                                    try:
                                        path_info = json.loads(diagram_path)
                                        if isinstance(path_info, dict) and "name" in path_info:
                                            actual_image_path = path_info["name"]
                                    except json.JSONDecodeError:
                                        if os.path.exists(diagram_path) and os.path.isfile(diagram_path):
                                            actual_image_path = diagram_path
                                elif isinstance(diagram_path, dict) and "name" in diagram_path:
                                    actual_image_path = diagram_path["name"]
                                else:
                                    if os.path.exists(diagram_path) and os.path.isfile(diagram_path):
                                        actual_image_path = diagram_path

                                # If we have a valid path, display the image
                                if actual_image_path and os.path.exists(actual_image_path):
                                    # Display thumbnail version
                                    st.image(actual_image_path, width=500)

                                    # Get original image dimensions
                                    img_width, img_height = Image.open(actual_image_path).size

                                    # Create a unique key for the fullscreen button
                                    button_key = f"fullscreen_{os.path.basename(actual_image_path)}"

                                    # Add a fullscreen button that emphasizes original size
                                    if st.button("View 100% Original Size in Fullscreen", key=button_key):
                                        # Convert image to base64 for embedding
                                        with open(actual_image_path, "rb") as img_file:
                                            encoded = base64.b64encode(img_file.read()).decode()

                                        # Create HTML that ensures the image is displayed at 100% original size
                                        html = f"""
                                        <html>
                                        <head>
                                            <title>Original Size Diagram</title>
                                            <style>
                                                body, html {{
                                                    margin: 0;
                                                    padding: 0;
                                                    height: 100%;
                                                    width: 100%;
                                                    background-color: #000;
                                                    overflow: auto;
                                                }}
                                                .controls {{
                                                    position: fixed;
                                                    top: 15px;
                                                    right: 15px;
                                                    z-index: 1000;
                                                    display: flex;
                                                    gap: 10px;
                                                }}
                                                .btn {{
                                                    background-color: rgba(255, 255, 255, 0.8);
                                                    border: none;
                                                    padding: 8px 15px;
                                                    border-radius: 4px;
                                                    cursor: pointer;
                                                    font-size: 14px;
                                                    font-weight: bold;
                                                }}
                                                .img-container {{
                                                    position: absolute;
                                                    top: 0;
                                                    left: 0;
                                                    width: 100%;
                                                    height: 100%;
                                                    display: flex;
                                                    justify-content: center;
                                                    align-items: center;
                                                    padding: 20px;
                                                    box-sizing: border-box;
                                                }}
                                                .img-wrapper {{
                                                    overflow: auto;
                                                    max-width: 100%;
                                                    max-height: 100%;
                                                }}
                                                #diagram {{
                                                    /* Ensure image displays at exact original size - no scaling */
                                                    width: {img_width}px;
                                                    height: {img_height}px;
                                                    image-rendering: pixelated;
                                                    image-rendering: -webkit-optimize-contrast;
                                                }}
                                                .size-info {{
                                                    position: fixed;
                                                    bottom: 15px;
                                                    left: 15px;
                                                    background-color: rgba(255, 255, 255, 0.8);
                                                    padding: 5px 10px;
                                                    border-radius: 4px;
                                                    font-size: 12px;
                                                }}
                                            </style>
                                        </head>
                                        <body>
                                            <div class="controls">
                                                <button id="zoomIn" class="btn">Zoom In (+)</button>
                                                <button id="zoomOut" class="btn">Zoom Out (-)</button>
                                                <button id="resetZoom" class="btn">Reset (100%)</button>
                                                <button id="toggleFullscreen" class="btn">Toggle Fullscreen</button>
                                                <button id="close" class="btn">Close</button>
                                            </div>

                                            <div class="img-container">
                                                <div class="img-wrapper">
                                                    <img id="diagram" src="data:image/png;base64,{encoded}" alt="Model Diagram">
                                                </div>
                                            </div>

                                            <div class="size-info">
                                                Original size: {img_width}×{img_height}px | Current zoom: <span id="zoomLevel">100%</span>
                                            </div>

                                            <script>
                                                // Variables to track current zoom level
                                                let currentZoom = 1.0;
                                                const zoomFactor = 0.1;
                                                const diagram = document.getElementById('diagram');
                                                const zoomLevelDisplay = document.getElementById('zoomLevel');

                                                // Function to update zoom
                                                function updateZoom() {{
                                                    diagram.style.width = `${{img_width * currentZoom}}px`;
                                                    diagram.style.height = `${{img_height * currentZoom}}px`;
                                                    zoomLevelDisplay.textContent = `${{Math.round(currentZoom * 100)}}%`;
                                                }}

                                                // Zoom in function
                                                document.getElementById('zoomIn').addEventListener('click', function() {{
                                                    currentZoom += zoomFactor;
                                                    updateZoom();
                                                }});

                                                // Zoom out function
                                                document.getElementById('zoomOut').addEventListener('click', function() {{
                                                    if (currentZoom > zoomFactor) {{
                                                        currentZoom -= zoomFactor;
                                                        updateZoom();
                                                    }}
                                                }});

                                                // Reset zoom function
                                                document.getElementById('resetZoom').addEventListener('click', function() {{
                                                    currentZoom = 1.0;
                                                    updateZoom();
                                                }});

                                                // Function to toggle fullscreen
                                                function toggleFullscreen() {{
                                                    if (!document.fullscreenElement &&
                                                        !document.mozFullScreenElement &&
                                                        !document.webkitFullscreenElement &&
                                                        !document.msFullscreenElement) {{
                                                        // Enter fullscreen
                                                        if (document.documentElement.requestFullscreen) {{
                                                            document.documentElement.requestFullscreen();
                                                        }} else if (document.documentElement.msRequestFullscreen) {{
                                                            document.documentElement.msRequestFullscreen();
                                                        }} else if (document.documentElement.mozRequestFullScreen) {{
                                                            document.documentElement.mozRequestFullScreen();
                                                        }} else if (document.documentElement.webkitRequestFullscreen) {{
                                                            document.documentElement.webkitRequestFullscreen(Element.ALLOW_KEYBOARD_INPUT);
                                                        }}
                                                    }} else {{
                                                        // Exit fullscreen
                                                        if (document.exitFullscreen) {{
                                                            document.exitFullscreen();
                                                        }} else if (document.msExitFullscreen) {{
                                                            document.msExitFullscreen();
                                                        }} else if (document.mozCancelFullScreen) {{
                                                            document.mozCancelFullScreen();
                                                        }} else if (document.webkitExitFullscreen) {{
                                                            document.webkitExitFullscreen();
                                                        }}
                                                    }}
                                                }}

                                                // Auto-enter fullscreen when the page loads
                                                document.addEventListener('DOMContentLoaded', function() {{
                                                    toggleFullscreen();
                                                }});

                                                // Set up event listeners
                                                document.getElementById('toggleFullscreen').addEventListener('click', toggleFullscreen);
                                                document.getElementById('close').addEventListener('click', function() {{
                                                    if (document.fullscreenElement) toggleFullscreen();
                                                    window.close(); // Try to close window
                                                    window.history.back(); // Fallback to going back
                                                }});
                                            </script>
                                        </body>
                                        </html>
                                        """

                                        # Open in a new browser tab
                                        from tempfile import NamedTemporaryFile
                                        import webbrowser

                                        with NamedTemporaryFile(delete=False, suffix='.html') as tmp:
                                            tmp.write(html.encode())
                                            webbrowser.open('file://' + tmp.name, new=2)
                                else:
                                    st.error(f"Diagram file not found: {diagram_path}")

                            except Exception as e:
                                st.error(f"Error displaying diagram: {str(e)}")
                                import traceback
                                st.code(traceback.format_exc())

                        # Display framework and architecture
                        if metadata.get('framework') or metadata.get('architecture'):
                            st.subheader("Technical Information")
                            tech_cols = st.columns(2)
                            with tech_cols[0]:
                                st.write("**Framework:**")
                                framework = metadata.get('framework', {})
                                if isinstance(framework, dict):
                                    st.write(f"{framework.get('name', 'N/A')} {framework.get('version', '')}")
                                else:
                                    st.write(str(framework))
                            with tech_cols[1]:
                                st.write("**Architecture:**")
                                arch = metadata.get('architecture', {})
                                if isinstance(arch, dict):
                                    st.write(f"{arch.get('type', 'N/A')}")
                                else:
                                    st.write(str(arch))

                        # Display dataset and training info
                        if metadata.get('dataset') or metadata.get('training_config'):
                            st.subheader("Training Information")
                            dataset = metadata.get('dataset', {})
                            if isinstance(dataset, dict):
                                st.write(f"**Dataset:** {dataset.get('name', 'N/A')}")

                            training = metadata.get('training_config', {})
                            if isinstance(training, dict):
                                train_cols = st.columns(4)
                                with train_cols[0]:
                                    st.metric("Batch Size", training.get('batch_size', 'N/A'))
                                with train_cols[1]:
                                    st.metric("Learning Rate", training.get('learning_rate', 'N/A'))
                                with train_cols[2]:
                                    st.metric("Optimizer", training.get('optimizer', 'N/A'))
                                with train_cols[3]:
                                    st.metric("Epochs", training.get('epochs', 'N/A'))

        # [Rest of the method remains unchanged]
        elif t == "image_search":
            st.header("Image Search Results")
            # Display images in a grid layout
            if "results" in r and len(r.get("results", [])) > 0:
                # Create columns for a grid layout
                num_cols = 3  # Number of columns in the grid
                cols = st.columns(num_cols)

                # Distribute images across columns
                for i, img_result in enumerate(r.get("results", [])):
                    col_idx = i % num_cols
                    with cols[col_idx]:
                        path = img_result.get("image_path", "")
                        if path:
                            st.image(path, caption=img_result.get("prompt", ""))
                            st.write(f"**ID:** {img_result.get('id', 'N/A')}")

                            # Add metadata as an expander
                            if img_result.get("metadata"):
                                with st.expander("Image Metadata"):
                                    for key, value in img_result.get("metadata", {}).items():
                                        if key not in ["image_path", "thumbnail_path", "prompt", "id"]:
                                            st.write(f"**{key}:** {value}")

        elif t == "command":
            st.header("Command Results")
            if isinstance(r.get("result"), dict):
                # Format dictionary results
                if "available_commands" in r["result"]:
                    st.subheader("Available Commands")
                    # Create columns for command categories
                    cols = st.columns(2)
                    with cols[0]:
                        st.write("**Query Commands:**")
                        st.write("• query - Search for model scripts or images")
                    with cols[1]:
                        st.write("**Management Commands:**")
                        st.write("• list-models - List available models")
                        st.write("• list-images - List available images")
                        st.write("• generate-notebook - Generate a notebook")

                elif "available_models" in r["result"]:
                    st.subheader("Available Models")

                    # Prepare data for Streamlit DataFrame
                    models_data = []
                    for i, model in enumerate(r["result"]["available_models"]):
                        # Format dates
                        created = model.get('creation_date', 'N/A')
                        if isinstance(created, str) and len(created) > 10:
                            created = created[:10]

                        modified = model.get('last_modified_date', 'N/A')
                        if isinstance(modified, str) and len(modified) > 10:
                            modified = modified[:10]

                        # Add to data collection
                        models_data.append({
                            "Rank": i + 1,
                            "Model ID": model.get('model_id', f'Model {i + 1}'),
                            "Created": created,
                            "Modified": modified,
                            "Path": model.get('absolute_path', 'N/A')
                        })

                    # Create and display the DataFrame
                    import pandas as pd
                    df = pd.DataFrame(models_data)
                    st.dataframe(df, use_container_width=True)

                    # Add a download button for the table
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download as CSV",
                        data=csv,
                        file_name="available_models.csv",
                        mime="text/csv"
                    )

                elif "available_images" in r["result"]:
                    st.subheader("Available Images")

                    # Create a grid layout for images
                    num_cols = 3  # Number of columns in the grid
                    cols = st.columns(num_cols)

                    # Distribute images across columns
                    for i, img in enumerate(r["result"]["available_images"]):
                        col_idx = i % num_cols
                        with cols[col_idx]:
                            st.image(img["thumbnail_path"], caption=img["prompt"])
                            st.write(f"**ID:** {img.get('id', 'N/A')}")

                    # Add a table view option
                    with st.expander("Table View"):
                        # Prepare data for Streamlit DataFrame
                        images_data = []
                        for i, img in enumerate(r["result"]["available_images"]):
                            images_data.append({
                                "ID": img.get('id', f'Image {i + 1}'),
                                "Prompt": img.get('prompt', 'N/A'),
                                "Path": img.get('image_path', 'N/A')
                            })

                        # Create and display the DataFrame
                        import pandas as pd
                        df = pd.DataFrame(images_data)
                        st.dataframe(df, use_container_width=True)

                else:
                    # Generic dictionary formatting with better UI
                    cols = st.columns(2)
                    for i, (key, value) in enumerate(r["result"].items()):
                        col_idx = i % 2
                        with cols[col_idx]:
                            st.metric(key, value)
            else:
                # Simple string result
                st.success(r.get("result", ""))

        elif t == "error":
            st.error(r["error"])

        elif t == "needs_clarification":
            # Handle query that needs clarification
            query = r.get("query", "")
            clarity_result = r.get("clarity_result", {})

            st.warning("Your query could be clearer")
            st.write(f"**Reason**: {clarity_result.get('reason', 'No reason provided')}")

            improved_query = clarity_result.get('improved_query', query)
            suggestions = clarity_result.get('suggestions', [])

            st.write("**Suggestions**:")
            suggestion_options = [f"Improved query: {improved_query}"] + suggestions + [f"Use original query: {query}",
                                                                                        "Enter a new query"]

            choice = st.selectbox("Select an option:", suggestion_options)

            if st.button("Use this query"):
                if choice == suggestion_options[0]:  # Improved query
                    new_query = improved_query
                elif choice == suggestion_options[-2]:  # Original query
                    new_query = query
                elif choice == suggestion_options[-1]:  # New query
                    new_query = st.session_state.get('new_query', '')
                else:
                    # One of the suggestions
                    new_query = suggestions[suggestion_options.index(choice) - 1]

                if new_query:
                    asyncio.run(self.process_query(new_query))


class CommandHandler:
    """Handles command processing and delegation to appropriate components for Streamlit."""

    def __init__(self, streamlit_interface):
        """
        Initialize CommandHandler with a reference to the StreamlitInterface.

        Args:
            streamlit_interface: Instance of StreamlitInterface that this handler will work with
        """
        self.interface = streamlit_interface
        self.rag = streamlit_interface.rag
        self.components = self.rag.components
        self.user_id = st.session_state.user_id

        # These will be accessed from the StreamlitInterface
        self.model_display = streamlit_interface.model_display_manager
        self.image_display = streamlit_interface.image_display_manager
        self.notebook_generator = streamlit_interface.notebook_generator

    def handle_command(self, cmd):
        """
        Process a user command and delegate to the appropriate handler.

        Args:
            cmd (str): The command string to process.

        Returns:
            dict: Command execution results
        """
        cmd = cmd.strip()

        # Basic command handling - matching CLI's CommandHandler
        if cmd.lower() == "help":
            return self._handle_help_command()
        elif cmd.lower() == "list-models":
            return self._handle_list_models_command()
        elif cmd.lower() == "list-image_processing":
            return self._handle_list_images_command()
        elif cmd.lower().startswith("generate-notebook"):
            return self._handle_generate_notebook_command(cmd)
        else:
            # For unknown commands, let the RAGSystem try to process it
            result = self.rag.execute_command(cmd)
            return result

    def _handle_help_command(self):
        """Display available commands and their descriptions."""
        commands = {
            "query": "Search for model scripts or image_processing",
            "list-models": "List available models",
            "list-image_processing": "List available image_processing",
            "generate-notebook": "Generate a Colab notebook for a model"
        }

        return {
            "type": "command",
            "result": {
                "available_commands": [f"{cmd_name}: {cmd_desc}" for cmd_name, cmd_desc in commands.items()]
            }
        }

    def _handle_list_models_command(self):
        """List all models accessible to the current user."""
        access_control = self.components["vector_db"]["access_control"]

        try:
            # Get models the user has access to
            available_models = access_control.get_accessible_models(self.user_id)
            return {
                "type": "command",
                "result": {
                    "available_models": available_models
                }
            }
        except Exception as e:
            return {
                "type": "error",
                "error": f"Error listing models: {str(e)}"
            }

    def _handle_list_images_command(self):
        """List all image_processing accessible to the current user."""
        try:
            access_control = self.components["vector_db"]["access_control"]

            # Get image_processing the user has access to
            try:
                available_images = access_control.get_accessible_images(self.user_id)
            except AttributeError:
                # Fallback: Get all image_processing if access control is not properly implemented
                st.warning("Access control not fully implemented. Showing all available image_processing.")
                chroma_manager = self.components["vector_db"]["chroma_manager"]
                # Use a safer method to get image_processing
                available_images = asyncio.run(self._get_all_images(chroma_manager))

            return {
                "type": "command",
                "result": {
                    "available_images": available_images
                }
            }
        except Exception as e:
            return {
                "type": "error",
                "error": f"Error listing image_processing: {str(e)}"
            }

    async def _get_all_images(self, chroma_manager):
        """
        Fallback method to get all image_processing when access control fails.

        Args:
            chroma_manager: The Chroma database manager

        Returns:
            List of image dictionaries
        """
        try:
            # Query for all image_processing in the generated_images collection
            results = await chroma_manager.get(
                collection_name="generated_images",
                include=["metadatas"],
                limit=100  # Set a reasonable limit
            )

            if results and "results" in results:
                images = []
                for item in results["results"]:
                    metadata = item.get("metadata", {})
                    images.append({
                        "id": item.get("id", "Unknown"),
                        "prompt": metadata.get("prompt", "No prompt"),
                        "image_path": metadata.get("image_path", "Not available"),
                        "thumbnail_path": metadata.get("thumbnail_path",
                                                       metadata.get("image_path", "Not available")),
                        "metadata": metadata
                    })
                return images
            return []
        except Exception as e:
            st.error(f"Error retrieving image_processing: {str(e)}")
            return []

    def _handle_generate_notebook_command(self, cmd):
        """Handle the generate-notebook command."""
        notebook_type = "evaluation"

        if cmd.lower() == "generate-notebook":
            model_id = st.text_input("Enter model ID")
            output_path = st.text_input(
                "Enter output path", value=f"./notebooks/{model_id}.ipynb")
        else:
            parts = cmd.split(maxsplit=3)
            model_id = parts[1] if len(parts) > 1 else ""

            if len(parts) > 2:
                if parts[2].startswith("--type=") or parts[2].startswith("-t="):
                    notebook_type = parts[2].split("=", 1)[1]
                else:
                    notebook_type = parts[2]

            if len(parts) > 3:
                if parts[3].startswith("--output=") or parts[3].startswith("-o="):
                    output_path = parts[3].split("=", 1)[1]
                else:
                    output_path = parts[3]
            else:
                output_path = f"./notebooks/{model_id}_{notebook_type}.ipynb"

        if not model_id:
            return {
                "type": "error",
                "error": "Model ID is required"
            }

        try:
            result = self.notebook_generator.generate_notebook(self.components, model_id, output_path)
            if result:
                return {
                    "type": "command",
                    "result": f"Notebook generated successfully: {result}"
                }
            else:
                return {
                    "type": "error",
                    "error": "Failed to generate notebook"
                }
        except Exception as e:
            return {
                "type": "error",
                "error": f"Error generating notebook: {str(e)}"
            }


def run_streamlit_app(components):
    """
    Run the Streamlit app with pre-initialized components

    Args:
        components: Dictionary containing initialized system components
    """
    # ── reuse the same interface across reruns ──
    if "interface" not in st.session_state:
        st.session_state.interface = StreamlitInterface()
    interface = st.session_state.interface

    # Create command handler once interface is logged in
    if "command_handler" not in st.session_state and st.session_state.get('logged_in', False):
        st.session_state.command_handler = CommandHandler(interface)

    # Sidebar for settings and controls
    with st.sidebar:
        st.header("System Login")

        # User ID input
        user_id = st.text_input("User ID", value=st.session_state.user_id)
        if user_id != st.session_state.user_id:
            st.session_state.user_id = user_id

        # System login
        if not st.session_state.logged_in:
            if st.button("Login System"):
                with st.spinner("Logging into system..."):
                    success = interface.login(components)
                    if success:
                        st.session_state.command_handler = CommandHandler(interface)
                        st.success("System logged in successfully")
                    else:
                        st.error("Failed to login system")
        else:
            st.success("System is logged in")

        # Command selection
        st.header("Commands")
        command = st.radio(
            "Select Command",
            ["query", "list-models", "list-images", "generate-notebook", "help"]
        )
        st.session_state.selected_command = command

        # Advanced options
        st.header("Advanced Options")
        st.checkbox("Enable Query Clarity Check", value=False, key="enable_clarity_check")
        st.checkbox("Enable Comparison Detection", value=True, key="enable_comparison_detection")

        # Debug mode
        show_logs = st.checkbox("Show Debug Logs", value=False)

    # Main content area
    if not st.session_state.logged_in:
        st.info("Please login to the system using the sidebar")
    else:
        # Handle different commands
        command_handler = st.session_state.command_handler

        if st.session_state.selected_command == "query":
            st.header("Search Models")
            query_text = st.text_area("Enter your query:", key="query_text", height=100)

            col1, col2 = st.columns([1, 6])
            with col1:
                process_btn = st.button("Search")

            if process_btn and query_text:
                with st.spinner("Processing query..."):
                    asyncio.run(interface.process_query(query_text))

        elif st.session_state.selected_command == "list-models":
            st.header("Available Models")
            if st.button("Refresh Model List"):
                with st.spinner("Loading models..."):
                    # Use CommandHandler like in CLI
                    result = command_handler._handle_list_models_command()
                    st.session_state.results = result

        elif st.session_state.selected_command == "list-images":
            st.header("Available Images")
            if st.button("Refresh Image List"):
                with st.spinner("Loading images..."):
                    # Use CommandHandler like in CLI
                    result = command_handler._handle_list_images_command()
                    st.session_state.results = result

        elif st.session_state.selected_command == "generate-notebook":
            st.header("Generate Notebook")
            model_id = st.text_input("Model ID")
            notebook_type = st.selectbox("Notebook Type", ["evaluation", "training", "inference"])
            output_path = st.text_input("Output Path", value=f"./notebooks/{model_id}_{notebook_type}.ipynb")

            if st.button("Generate Notebook") and model_id:
                with st.spinner("Generating notebook..."):
                    # Use CommandHandler to process the command
                    cmd = f"generate-notebook {model_id} {notebook_type} {output_path}"
                    result = command_handler._handle_generate_notebook_command(cmd)
                    st.session_state.results = result

        elif st.session_state.selected_command == "help":
            st.header("Help")
            if st.button("Show Help"):
                result = command_handler._handle_help_command()
                st.session_state.results = result

        # Display current status
        status = st.session_state.status
        if status in ["processing", "searching", "processing_results", "generating_response"]:
            st.info(f"Status: {status.replace('_', ' ').title()}")
        elif status == "error":
            st.error("Error occurred")

        # Display results
        interface.render()

        # Display logs if debug mode is enabled
        if show_logs and st.session_state.logs:
            st.header("Debug Logs")
            for timestamp, log in reversed(st.session_state.logs[-20:]):
                # Format log as in CLIInterface
                if isinstance(log, dict):
                    level = log.get("level", "info")
                    message = log.get("message", str(log))
                else:
                    level = "info"
                    message = str(log)

                log_time = time.strftime("%H:%M:%S", time.localtime(timestamp))

                if level == "error":
                    st.error(f"[{log_time}] {message}")
                elif level == "warning":
                    st.warning(f"[{log_time}] {message}")
                else:
                    st.text(f"[{log_time}] {message}")