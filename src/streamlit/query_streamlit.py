import asyncio
import logging
import re
import time

from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode

import streamlit as st
from src.cli.cli_response_utils.image_display_manager import ImageDisplayManager
from src.cli.cli_response_utils.llm_response_processor import LLMResponseProcessor
from src.cli.cli_response_utils.model_display_manager import ModelDisplayManager
from src.core.notebook_generator import NotebookGenerator
from src.core.rag_system import RAGSystem
from src.core.base_command_handler import BaseCommandHandler

logging.basicConfig(level=logging.INFO)
import os
import json
import base64
from PIL import Image


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

    async def process_query(self, q, enable_clarity_check: bool = True):
        if not st.session_state.logged_in:
            st.error("Not logged in")
            return
        await self.rag.process_query(q,
                                     enable_clarity_check=enable_clarity_check,
                                     enable_comparison_detection=st.session_state.get('enable_comparison_detection',
                                                                                      True)
                                     )

    def render(self):
        """
        Display results in a streamlit-appropriate way
        Using AgGrid for clickable search results, with safe emptiness checks,
        and moving the description into the details' pane.
        """
        import pandas as pd

        r = st.session_state.results
        if not r:
            return
        t = r.get("type", "")

        if t == "text_search":
            st.header("Answer")

            # ——— LLM response with thinking tags ———
            response_text = r["final_response"]
            thinking_pattern = re.compile(r'<(thinking|think)>(.*?)</\1>', re.DOTALL)
            thinking_matches = thinking_pattern.findall(response_text)
            if thinking_matches:
                for tag_type, thinking_text in thinking_matches:
                    st.markdown(
                        f"<div style='background-color:#e6f3ff; padding:10px; border-radius:5px; "
                        f"margin-bottom:10px; color:#003366;'><strong>Thinking:</strong><br>{thinking_text}</div>",
                        unsafe_allow_html=True,
                    )
                clean_response = thinking_pattern.sub("", response_text)
                st.write(clean_response)
            else:
                st.write(response_text)

            # ——— clickable search results (no Description column) ———
            if "search_results" in r and r["search_results"]:
                st.subheader("Search Results")

                rows = []
                for i, result in enumerate(r["search_results"]):
                    meta = result.get("metadata", {}) or {}
                    # parse nested file JSON if needed
                    if isinstance(meta.get("file"), str):
                        try:
                            file_meta = json.loads(meta["file"])
                        except:
                            file_meta = {}
                    else:
                        file_meta = meta.get("file", {})

                    rows.append({
                        "Rank": i + 1,
                        "Model ID": result.get("model_id", result.get("id", f"Item {i+1}")),
                        "Created At": meta.get("created_at", "N/A"),
                        "Path": file_meta.get("absolute_path", meta.get("absolute_path", "N/A")),
                    })

                df = pd.DataFrame(rows)

                gb = GridOptionsBuilder.from_dataframe(df)
                gb.configure_selection(selection_mode="single", use_checkbox=False)
                grid_opts = gb.build()

                grid_resp = AgGrid(
                    df,
                    gridOptions=grid_opts,
                    update_mode=GridUpdateMode.SELECTION_CHANGED,
                    data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                    fit_columns_on_grid_load=True,
                    height=300,
                )

                # ——— robust selection check ———
                sel = None
                sel_rows = grid_resp.get("selected_rows", [])
                if isinstance(sel_rows, pd.DataFrame):
                    if not sel_rows.empty:
                        sel = sel_rows.iloc[0].to_dict()
                        idx = int(sel["Rank"]) - 1
                elif isinstance(sel_rows, list) and sel_rows:
                    sel = sel_rows[0]
                    idx = int(sel["Rank"]) - 1

                if sel is not None:
                    model = r["search_results"][idx]
                    meta = model.get("metadata", {}) or {}
                    # parse JSON subfields
                    for fld in ["file", "framework", "architecture", "dataset", "training_config"]:
                        if isinstance(meta.get(fld), str):
                            try:
                                meta[fld] = json.loads(meta[fld])
                            except:
                                pass

                    # ——— Details expander, auto-expanded ———
                    with st.expander(
                        f"Details for Rank {sel['Rank']} – {sel['Model ID']}",
                        expanded=True
                    ):
                        # —— Description moved here ——
                        st.subheader("Description")
                        desc = model.get("merged_description", "")
                        st.write(desc if desc else "No description available.")

                        # —— File Information ——
                        if meta.get("file"):
                            st.subheader("File Information")
                            f = meta["file"]
                            c1, c2, c3 = st.columns(3)
                            with c1:
                                size_kb = int(f.get("size_bytes", 0) / 1024)
                                st.metric("Size", f"{size_kb} KB")
                            with c2:
                                created = f.get("creation_date", "")[:10]
                                st.metric("Created", created or "N/A")
                            with c3:
                                modified = f.get("last_modified_date", "")[:10]
                                st.metric("Modified", modified or "N/A")

                        # —— Model Component Diagram ——
                        if meta.get("diagram_path"):
                            st.subheader("Model Component Diagram")
                            diagram_path = meta["diagram_path"]
                            try:
                                # resolve actual path
                                if isinstance(diagram_path, str) and diagram_path.startswith("{"):
                                    info = json.loads(diagram_path)
                                    actual = info.get("name")
                                elif isinstance(diagram_path, dict):
                                    actual = diagram_path.get("name")
                                else:
                                    actual = diagram_path

                                if not os.path.exists(actual):
                                    st.error(f"Diagram file not found: {diagram_path}")
                                else:
                                    img = Image.open(actual)
                                    w, h = img.size
                                    # thumbnail
                                    st.image(actual, width=500)
                                    # prepare base64
                                    with open(actual, "rb") as fh:
                                        encoded = base64.b64encode(fh.read()).decode()

                                    key = f"fullres_{os.path.basename(actual)}"
                                    if st.button("View at 100% Original Size", key=key):
                                        html_code = f"""
                                        <html>
                                          <body style="
                                            margin:0; padding:0;
                                            background-color:#000;
                                            display:flex; justify-content:center; align-items:center;
                                          ">
                                            <img
                                              src="data:image/png;base64,{encoded}"
                                              style="
                                                width:{w}px; height:{h}px;
                                                image-rendering:pixelated;
                                              "
                                            />
                                          </body>
                                        </html>
                                        """
                                        st.components.v1.html(
                                            html_code,
                                            height=min(h, 800),
                                            width=min(w, 1200),
                                            scrolling=True
                                        )
                            except Exception as e:
                                st.error(f"Error displaying diagram: {e}")

                        # —— Technical Information ——
                        if meta.get("framework") or meta.get("architecture"):
                            st.subheader("Technical Information")
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.write("**Framework:**")
                                fw = meta.get("framework", {})
                                if isinstance(fw, dict):
                                    st.write(f"{fw.get('name','N/A')} {fw.get('version','')}")
                                else:
                                    st.write(fw)
                            with col_b:
                                st.write("**Architecture:**")
                                arch = meta.get("architecture", {})
                                if isinstance(arch, dict):
                                    st.write(arch.get("type","N/A"))
                                else:
                                    st.write(arch)

                        # —— Training Information ——
                        if meta.get("dataset") or meta.get("training_config"):
                            st.subheader("Training Information")
                            ds = meta.get("dataset", {})
                            if isinstance(ds, dict):
                                st.write(f"**Dataset:** {ds.get('name','N/A')}")
                            tc = meta.get("training_config", {})
                            if isinstance(tc, dict):
                                tcols = st.columns(4)
                                specs = [
                                    ("Batch Size", "batch_size"),
                                    ("Learning Rate", "learning_rate"),
                                    ("Optimizer", "optimizer"),
                                    ("Epochs", "epochs"),
                                ]
                                for col, (lbl, key) in zip(tcols, specs):
                                    col.metric(lbl, tc.get(key, "N/A"))

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

            # Initialize clarification state if not present
            if "clarification_state" not in st.session_state:
                st.session_state.clarification_state = {
                    "query": query,
                    "improved_query": clarity_result.get('improved_query', query),
                    "suggestions": clarity_result.get('suggestions', []),
                    "reason": clarity_result.get('reason', 'No reason provided'),
                    "unique_id": str(time.time())  # Create a timestamp-based unique ID
                }

            # Get state from session
            cstate = st.session_state.clarification_state

            unique_id = cstate["unique_id"]
            st.warning("Your query could be clearer")
            st.write(f"**Reason**: {cstate['reason']}")
            st.write("**Suggestions**:")

            suggestion_options = [f"Improved query: {cstate['improved_query']}"] + cstate['suggestions'] + [
                f"Use original query: {cstate['query']}", "Enter a new query"
            ]

            # Use the stable unique_id for all widget keys
            choice = st.selectbox("Select an option:", suggestion_options, key=f"clarification_options_{unique_id}")

            # Create a text area for new query input that's shown conditionally
            new_query_input = ""
            if choice == "Enter a new query":
                new_query_input = st.text_area("Enter new query:", key=f"new_query_input_{unique_id}", height=100)

            if st.button("Use this query", key=f"use_query_button_{unique_id}"):
                # Clear clarification state when a query is selected
                if choice == suggestion_options[0]:  # Improved query
                    new_query = cstate['improved_query']

                elif choice == suggestion_options[-2]:  # Original query
                    new_query = cstate['query']

                elif choice == suggestion_options[-1]:  # New query
                    new_query = new_query_input

                else:
                    # One of the suggestions
                    new_query = cstate['suggestions'][suggestion_options.index(choice) - 1]

                if new_query:
                    # Clear the clarification state to prevent re-displaying
                    if "clarification_state" in st.session_state:
                        del st.session_state.clarification_state

                    with st.spinner("Processing query..."):
                        asyncio.run(self.process_query(new_query, False))

                        # Display current status
                        status = st.session_state.status
                        if status in ["processing", "searching", "processing_results", "generating_response"]:
                            st.info(f"Status: {status.replace('_', ' ').title()}")

                        elif status == "error":
                            st.error("Error occurred")

                        # Display results
                        self.render()

class CommandHandler(BaseCommandHandler):
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

        super().__init__(self.rag, self.components, self.user_id)

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
        try:
            available_models = self.list_models()
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
            available_images = self.list_images()
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
            result = self.generate_notebook(model_id, output_path)
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


class StreamlitApp:
    def __init__(self, components: dict):
        """
        Initialize the StreamlitApp wrapper with the provided system components.
        """
        self.components = components

    def run(self):
        """
        Main entry point for the Streamlit app. Manages session state,
        renders the sidebar, and routes to appropriate command handlers.
        """
        self._ensure_interface()
        self._ensure_command_handler()
        self._render_sidebar()

        if not st.session_state.logged_in:
            st.info("Please login to the system using the sidebar")
        else:
            self._handle_main_area()

        # Render results, status, and logs as needed
        st.session_state.interface.render()
        self._render_debug_logs()

    def _ensure_interface(self):
        """
        Ensure a single StreamlitInterface instance is stored in session_state.
        """
        if "interface" not in st.session_state:
            st.session_state.interface = StreamlitInterface()

    def _ensure_command_handler(self):
        """
        Once logged in, create a CommandHandler tied to the interface.
        """
        if (
            "command_handler" not in st.session_state
            and st.session_state.get("logged_in", False)
        ):
            st.session_state.command_handler = CommandHandler(st.session_state.interface)

    def _render_sidebar(self):
        """
        Render the sidebar, including login controls, command selection,
        and advanced options.
        """
        with st.sidebar:
            st.header("System Login")
            self._sidebar_user_id_input()
            self._sidebar_login_button()

            st.header("Commands")
            st.radio(
                "Select Command",
                ["query", "list-models", "list-images", "generate-notebook", "help"],
                key="selected_command",
            )

            st.header("Advanced Options")
            st.checkbox(
                "Enable Query Clarity Check", value=True, key="enable_clarity_check"
            )
            st.checkbox(
                "Enable Comparison Detection", value=True, key="enable_comparison_detection"
            )

            st.checkbox("Show Debug Logs", value=False, key="show_logs")

    def _sidebar_user_id_input(self):
        """
        Render the User ID input text box and update session_state if changed.
        """
        user_id = st.text_input("User ID", value=st.session_state.get("user_id", ""))
        if user_id != st.session_state.get("user_id", ""):
            st.session_state.user_id = user_id

    def _sidebar_login_button(self):
        """
        If not already logged in, show a Login button. If clicked, attempt login.
        """
        if not st.session_state.get("logged_in", False):
            if st.button("Login System"):
                with st.spinner("Logging into system..."):
                    success = st.session_state.interface.login(self.components)
                    if success:
                        st.session_state.command_handler = CommandHandler(
                            st.session_state.interface
                        )
                        st.session_state.logged_in = True
                        st.success("System logged in successfully")
                    else:
                        st.error("Failed to login system")
        else:
            st.success("System is logged in")

    def _handle_main_area(self):
        """
        Render the main content area based on the selected command.
        """
        command = st.session_state.selected_command
        handler = st.session_state.command_handler

        if command == "query":
            self._handle_query(handler)
        elif command == "list-models":
            self._handle_list_models(handler)
        elif command == "list-images":
            self._handle_list_images(handler)
        elif command == "generate-notebook":
            self._handle_generate_notebook(handler)
        elif command == "help":
            self._handle_help(handler)

        self._render_status()

    def _handle_query(self, handler: "CommandHandler"):
        """
        Process the 'query' command: show a text area and Search button.
        """
        st.header("Search Models")
        query_text = st.text_area("Enter your query:", key="query_text", height=100)
        col1, _ = st.columns([1, 6])
        with col1:
            process_btn = st.button("Search")

        if process_btn and query_text:
            with st.spinner("Processing query..."):
                asyncio.run(
                    handler._handle_query_command(
                        query_text, st.session_state.get("enable_clarity_check", True)
                    )
                )

    def _handle_list_models(self, handler: "CommandHandler"):
        """
        Process the 'list-models' command: Refresh Model List button.
        """
        st.header("Available Models")
        if st.button("Refresh Model List"):
            with st.spinner("Loading models..."):
                result = handler._handle_list_models_command()
                st.session_state.results = result

    def _handle_list_images(self, handler: "CommandHandler"):
        """
        Process the 'list-images' command: Refresh Image List button.
        """
        st.header("Available Images")
        if st.button("Refresh Image List"):
            with st.spinner("Loading images..."):
                result = handler._handle_list_images_command()
                st.session_state.results = result

    def _handle_generate_notebook(self, handler: "CommandHandler"):
        """
        Process the 'generate-notebook' command: collect model_id,
        notebook_type, and output_path from user inputs and invoke handler.
        """
        st.header("Generate Notebook")
        model_id = st.text_input("Model ID", key="nb_model_id")
        notebook_type = st.selectbox(
            "Notebook Type", ["evaluation", "training", "inference"], key="nb_type"
        )
        default_path = f"./notebooks/{model_id}_{notebook_type}.ipynb"
        output_path = st.text_input("Output Path", value=default_path, key="nb_output")

        if st.button("Generate Notebook") and model_id:
            with st.spinner("Generating notebook..."):
                cmd = f"generate-notebook {model_id} {notebook_type} {output_path}"
                result = handler._handle_generate_notebook_command(cmd)
                st.session_state.results = result

    def _handle_help(self, handler: "CommandHandler"):
        """
        Process the 'help' command: show help when button is clicked.
        """
        st.header("Help")
        if st.button("Show Help"):
            result = handler._handle_help_command()
            st.session_state.results = result

    def _render_status(self):
        """
        Display the current system status (processing, error, etc.).
        """
        status = st.session_state.get("status", "")
        if status in [
            "processing",
            "searching",
            "processing_results",
            "generating_response",
        ]:
            st.info(f"Status: {status.replace('_', ' ').title()}")
        elif status == "error":
            st.error("Error occurred")

    def _render_debug_logs(self):
        """
        If debug mode is enabled, display the last 20 log entries.
        """
        if st.session_state.get("show_logs", False):
            logs = st.session_state.get("logs", [])
            if not logs:
                return

            st.header("Debug Logs")
            for timestamp, log in reversed(logs[-20:]):
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


def run_streamlit_app(components: dict):
    """
    Run the Streamlit app with pre-initialized components.

    Args:
        components: Dictionary containing initialized system components
    """
    app = StreamlitApp(components)
    app.run()