"""
RAG System Core Module - Encapsulating CLI functionality into callable functions

This module provides the core functionality of the RAG system, separating business logic from the UI layer.
All core functionality is exposed through the RAGSystem class, which can be called by any interface (CLI, API, etc.).
"""

import asyncio
import logging
from typing import Dict, Any, Callable, List, Tuple

def remove_thinking_sections(text: str) -> str:
    """
    Remove all <thinking>...</thinking> sections from the input text.

    Args:
        text: The input string containing zero or more <thinking> sections.

    Returns:
        A new string with all <thinking> sections and their content removed.
    """
    import re
    # Regex to match <thinking>...</thinking> including multiline content (non-greedy)
    pattern = re.compile(r'<thinking>.*?</thinking>', re.DOTALL)
    # Replace matched sections with an empty string
    return pattern.sub('', text)

def generate_report_prompt_extensions():
    thinking_process_requirements = (
        "### THINKING PROCESS REQUIREMENTS\n\n"
        "Before constructing each report, engage in thorough analytical reasoning enclosed in <thinking></thinking> tags that demonstrates:\n"
        "- Carefully analyzes the user's query to identify core information needs and technical context\n"
        "- Systematically evaluates all provided search results from most to least relevant\n"
        "- Identifies connections, patterns, and relationships between different data points\n"
        "- Considers multiple possible interpretations of the technical data\n"
        "- Distinguishes between explicit facts and implied conclusions\n"
        "- Recognizes information gaps and acknowledges limitations in the available data\n"
        "- Prioritizes information based on relevance to the specific ML engineering context\n"
        "- Builds a coherent mental model of the ML system being described\n"
        "- Formulates insights that would be valuable for implementation or reproduction\n\n"
        "Your thinking should progress from initial observations to deeper technical understanding, "
        "questioning assumptions and validating conclusions as you proceed. Document this process "
        "transparently, showing how your understanding evolves based on the evidence provided."
    )

    report_structure_requirements = (
        "### REPORT STRUCTURE REQUIREMENTS\n\n"
        "Structure your technical ML report with the following cli_response_utils in a logical flow:\n\n"
        "1. **Executive Summary**\n"
        "   - Brief overview of the ML system or component being described\n"
        "   - Key technical characteristics and distinguishing features\n"
        "   - Primary findings derived from the search results\n\n"

        "2. **Technical Specifications**\n"
        "   - Detailed breakdown of architecture, cli_response_utils, and configurations\n"
        "   - Clear presentation of relevant parameters, hyperparameters, or settings\n"
        "   - Explicit citation of data sources for each technical specification\n\n"

        "3. **Implementation Details**\n"
        "   - Critical procedures, methodologies, or algorithms employed\n"
        "   - Technical workflows with step-by-step breakdowns where appropriate\n"
        "   - Environment or infrastructure requirements if specified\n\n"

        "4. **Performance Analysis**\n"
        "   - Quantitative metrics and evaluation results\n"
        "   - Comparative analysis or benchmarking if available\n"
        "   - Critical assessment of strengths and limitations\n\n"

        "5. **Technical Insights**\n"
        "   - Synthesis of key technical findings across search results\n"
        "   - Identification of design principles or patterns in the implementation\n"
        "   - Analysis of trade-offs or engineering decisions\n\n"

        "6. **Reproduction Guidance**\n"
        "   - Essential information for ML engineers to reproduce the system\n"
        "   - Potential challenges or considerations for implementation\n"
        "   - Extension or optimization opportunities if apparent\n\n"

        "7. **Information Gaps**\n"
        "   - Explicit acknowledgment of missing critical information\n"
        "   - Identification of areas that would benefit from additional data\n\n"

        "Adapt this structure as appropriate to the specific query and available information, "
        "expanding sections with substantial data and condensing or omitting those without sufficient support."
    )

    style_and_format_guidance = (
        "### STYLE AND FORMAT GUIDANCE\n\n"
        "Format your ML technical report according to these principles:\n\n"

        "**Technical Precision**\n"
        "- Define ALL technical terms upon first use\n"
        "- Use precise, unambiguous technical language\n"
        "- Maintain mathematical and statistical accuracy\n"
        "- Present numerical data with appropriate units and precision\n"
        "- Use consistent technical terminology throughout\n\n"

        "**Clarity and Accessibility**\n"
        "- Structure information with clear headers and subheaders\n"
        "- Use concise paragraphs with single main ideas\n"
        "- Employ bullet points for lists of features, parameters, or specifications\n"
        "- Use tables for structured parameter comparisons when appropriate\n"
        "- Include visual representations or pseudocode when it enhances understanding\n\n"

        "**Citation and Evidence**\n"
        "- Cite specific search results for all technical claims\n"
        "- Differentiate between direct citations and derived insights\n"
        "- Maintain transparent reasoning chains from evidence to conclusions\n"
        "- Clearly indicate when information spans multiple sources\n"
        "- Note explicitly when making comparisons or connections not stated in original sources\n\n"

        "**Objectivity and Completeness**\n"
        "- Present balanced analysis of strengths and limitations\n"
        "- Avoid excessive technical jargon without explanation\n"
        "- Acknowledge uncertainty where appropriate\n"
        "- Ensure no critical conceptual gaps in explanations\n"
        "- Bridge between theoretical concepts and practical implementation\n\n"

        "The report should read as if written by a senior ML architect with deep technical knowledge "
        "and practical implementation experience, balancing theoretical understanding with "
        "pragmatic engineering considerations."
    )

    combined_meta_prompt_extensions = (
        f"{thinking_process_requirements}\n\n"
        f"{report_structure_requirements}\n\n"
        f"{style_and_format_guidance}\n\n"
    )

    return combined_meta_prompt_extensions

class RAGSystem:
    """
    RAG System Core Class - Encapsulating all core functionality into callable functions

    This class provides all core functionality of the RAG system, including:
    - Initializing system cli_response_utils
    - Processing user queries
    - Generating responses
    - Handling model and image display

    By encapsulating these functionalities in a single class, we can easily reuse them in different cli_interface (CLI, Web, etc.).
    """

    def __init__(self):
        """Initialize RAG system core cli_response_utils"""
        self.components = None
        self.user_id = "anonymous"
        self.callbacks = {
            "on_log": lambda msg: None,  # Default log callback
            "on_result": lambda result: None,  # Default result callback
            "on_error": lambda error: None,  # Default error callback
            "on_status": lambda status: None,  # Default status callback
        }

    def initialize(self, components: Dict[str, Any], user_id: str = "anonymous") -> bool:
        """
        Initialize RAG system cli_response_utils

        Args:
            components: Dictionary containing initialized system cli_response_utils
            user_id: User ID, default is "anonymous"

        Returns:
            bool: Whether initialization was successful
        """
        try:
            self.components = components
            self.user_id = user_id

            # Record initialization status
            self._log(f"RAG system core initialized, user ID: {user_id}")
            self._update_status("ready")

            return True
        except Exception as e:
            self._log(f"Initialization failed: {str(e)}", level="error")
            self._handle_error(e)
            return False

    def register_callback(self, event_type: str, callback: Callable) -> None:
        """
        Register callback function

        Args:
            event_type: Event type ('on_log', 'on_result', 'on_error', 'on_status')
            callback: Callback function
        """
        if event_type in self.callbacks:
            self.callbacks[event_type] = callback
        else:
            self._log(f"Unknown event type: {event_type}", level="warning")

    async def process_query(self, query_text: str, enable_comparison_detection: bool = True) -> Dict[str, Any]:
        """
        Process user query and return results

        This method encapsulates the process_query functionality from the original CLI,
        but returns results as a callable function instead of printing directly to the console.

        Args:
            query_text: User query text
            enable_comparison_detection: Whether to enable comparison detection and processing

        Returns:
            Dict: Dictionary containing query results
        """
        try:
            self._log(f"Processing query: {query_text}")
            self._update_status("processing")

            # Ensure components are initialized
            if not self.components:
                raise ValueError("System components not initialized")

            # Extract required components
            llm_interface = self.components["response_generator"]["llm_interface"]

            # Detect if this is a comparison query (only if enabled)
            if enable_comparison_detection:
                is_comparison, retrieval_queries = await self._detect_comparison_query(query_text, llm_interface)

                print(f"retrieval_queries: {retrieval_queries}")

                if is_comparison and retrieval_queries:
                    return await self._process_comparison_query(query_text, retrieval_queries, llm_interface)

            # If not a comparison query or comparison detection is disabled, process as a regular query
            return await self._process_regular_query(query_text)

        except Exception as e:
            self._log(f"Query processing failed: {str(e)}", level="error")
            self._update_status("error")
            self._handle_error(e)

            return {
                "type": "error",
                "query": query_text,
                "error": str(e)
            }

    async def _process_comparison_query(self, query_text: str, retrieval_queries: List[str], llm_interface, max_output_counts: int = 4) -> Dict[
        str, Any]:
        """
        Process a comparison query by handling multiple retrieval queries and generating a comparison response.

        Args:
            query_text: The original comparison query
            retrieval_queries: List of retrieval queries to process
            llm_interface: Interface to the LLM

        Returns:
            Dict: The comparison query result
        """
        self._log(f"Processing comparison query with {len(retrieval_queries)} retrieval queries")
        self._update_status("processing_comparison")

        # Process each retrieval query separately and collect results
        comparison_results = []
        for sub_query in retrieval_queries:
            self._log(f"Processing sub-query: {sub_query}")
            # Process the sub-query but skip LLM response generation
            sub_result = await self._process_regular_query(sub_query, generate_llm_response=False, max_output_counts=(max_output_counts // len(retrieval_queries)))
            comparison_results.append(sub_result)

        # Once all retrieval queries are processed, generate a comparison response
        self._update_status("generating_comparison")
        comparison_response = await self._generate_comparison_response(
            query_text, comparison_results, llm_interface
        )

        result = {
            "type": "comparison_search",
            "query": query_text,
            "sub_queries": retrieval_queries,
            "sub_results": comparison_results,
            "final_response": comparison_response
        }

        self._update_status("completed")
        self._handle_result(result)

        print(f"result: {result['final_response']}")

        return result

    async def _process_regular_query(self, query_text: str, generate_llm_response: bool = True, max_output_counts: int = 3) -> Dict[str, Any]:
        """
        Process a regular (non-comparison) query.

        Args:
            query_text: The query text to process
            generate_llm_response: Whether to generate an LLM response or just do retrieval/ranking

        Returns:
            Dict: The query result
        """
        # Extract required components
        query_parser = self.components["query_engine"]["query_parser"]
        search_dispatcher = self.components["query_engine"]["search_dispatcher"]
        query_analytics = self.components["query_engine"]["query_analytics"]
        reranker = self.components["query_engine"]["reranker"]
        llm_interface = self.components["response_generator"]["llm_interface"]

        # Parse query
        parsed_query = query_parser.parse_query(query_text)
        self._log(f"Parse result: {parsed_query}")

        # Log query
        query_analytics.log_query(self.user_id, query_text, parsed_query)

        self._update_status("searching")

        # Dispatch query
        search_results = await search_dispatcher.dispatch(
            query=parsed_query.get("processed_query", query_text),
            intent=parsed_query["intent"],
            parameters=parsed_query["parameters"],
            user_id=self.user_id
        )

        # Process search results
        self._update_status("processing_results")

        if parsed_query["intent"] == "image_search":
            # Image search processing
            result = {
                "type": "image_search",
                "results": search_results,
                "query": query_text,
                "parsed_query": parsed_query
            }
        else:
            # Regular search processing
            reranked_results = self._process_search_results(
                search_results, reranker, parsed_query, query_text, max_to_return=max_output_counts, rerank_threshold=0
            )

            # Remove content field as it's not needed for display
            reranked_results = self._remove_field(reranked_results, "content")

            # Build results text
            results_text = self._build_results_text(reranked_results)

            # Initialize result structure
            result = {
                "type": "text_search" if generate_llm_response else "retrieval_only",
                "query": query_text,
                "parsed_query": parsed_query,
                "search_results": reranked_results,
                "results_text": results_text  # Include formatted results text
            }

            # Generate LLM response if requested
            if generate_llm_response:
                # Build meta prompt
                meta_prompt = self._build_meta_prompt(query_text, reranked_results, llm_interface)
                print(f"Meta prompt: {meta_prompt}\n")
                result["meta_prompt"] = meta_prompt

                # Build system prompt
                system_prompt = meta_prompt + "\n\n" + generate_report_prompt_extensions() + \
                                f"You are provided user query and its searched results.\n"

                # Build user prompt
                user_prompt = f"\nUser query:\n{query_text}\n"
                user_prompt += f"Search results:\n{results_text}"

                # Generate final response
                self._update_status("generating_response")
                final_response = llm_interface.generate_structured_response(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=0.25,
                    max_tokens=32000
                )

                # Process result
                if isinstance(final_response, dict) and 'content' in final_response:
                    content = final_response['content']
                    content = remove_thinking_sections(content)
                else:
                    content = str(final_response)
                    content = remove_thinking_sections(content)

                result["final_response"] = content
            else:
                # Skip LLM response generation
                result["final_response"] = ""

        if generate_llm_response:
            self._update_status("completed")
            self._handle_result(result)

        return result

    async def _detect_comparison_query(self, query_text: str, llm_interface) -> Tuple[bool, List[str]]:
        """
        Detect if a query is asking for a comparison and generate retrieval queries.

        Args:
            query_text: The original user query
            llm_interface: Interface to the LLM

        Returns:
            Tuple[bool, List[str]]: (is_comparison, list_of_retrieval_queries)
        """
        system_prompt = (
            "You are an expert AI assistant tasked with analyzing search queries. "
            "Your task is to determine if a query is asking for a comparison between two or more entities, "
            "and if so, to break it down into separate 'Find' retrieval queries."
            "\n\n"
            "Follow these steps:\n"
            "1. Determine if the query is asking for a comparison (e.g., differences, similarities, contrasts).\n"
            "2. If it IS a comparison query, identify the entities being compared.\n"
            "3. For each entity, create a separate retrieval query that starts with 'Find' followed by the entity and its relevant attributes.\n"
            "4. Keep the retrieval queries concise and focused on the specific entity.\n"
            "5. If it is NOT a comparison query, return 'false' and an empty list.\n"
            "\n"
            "Format your response as a JSON object with two fields:\n"
            "- 'is_comparison': true or false\n"
            "- 'retrieval_queries': a list of strings, each being a separate 'Find' retrieval query"
        )

        user_prompt = f"Analyze this query: {query_text}"

        response = llm_interface.generate_structured_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.05,
            max_tokens=3000
        )

        try:
            # Parse the JSON response
            if isinstance(response, dict) and 'content' in response:
                content = response['content']
            else:
                content = str(response)

            import json
            import re

            # Extract JSON from the response (it might be wrapped in markdown code blocks)
            json_pattern = r'```json\s*(.*?)\s*```'
            match = re.search(json_pattern, content, re.DOTALL)
            if match:
                json_str = match.group(1)
            else:
                # If not in code blocks, try to find the JSON directly
                json_pattern = r'({.*})'
                match = re.search(json_pattern, content, re.DOTALL)
                if match:
                    json_str = match.group(1)
                else:
                    json_str = content

            result = json.loads(json_str)
            return result.get('is_comparison', False), result.get('retrieval_queries', [])
        except Exception as e:
            self._log(f"Error parsing comparison detection response: {str(e)}", level="error")
            return False, []

    async def _generate_comparison_response(self, original_query: str, comparison_results: List[Dict],
                                            llm_interface) -> str:
        """
        Generate a comparison response based on multiple query results.

        Args:
            original_query: The original comparison query
            comparison_results: List of results from individual retrieval queries
            llm_interface: Interface to the LLM

        Returns:
            str: The generated comparison response
        """
        system_prompt = (
            "You are a senior machine learning architect with expertise in creating clear technical comparisons. "
            "Your task is to synthesize multiple search results into a comprehensive comparison "
            "that addresses the user's original query."
            "\n\n"
            "### THINKING PROCESS REQUIREMENTS\n\n"
            "Before constructing your comparison, engage in thorough analytical reasoning enclosed in <thinking></thinking> tags that demonstrates:\n"
            "- Carefully analyzing the original query to identify what aspects need comparison\n"
            "- Systematically evaluating all provided search results for each entity\n"
            "- Identifying significant points of comparison across the entities\n"
            "- Organizing comparison points by technical relevance and importance\n"
            "- Distinguishing between explicit facts and implied conclusions\n"
            "- Recognizing information gaps and acknowledging limitations\n"
            "- Building progressively deeper insights about the compared entities\n"
            "- Finding patterns and connections that might not be immediately obvious\n"
            "- Considering alternative interpretations of the technical differences\n\n"

            "Your thinking should flow naturally and organically, demonstrating genuine discovery and insight "
            "rather than mechanical analysis. Start with basic observations, develop deeper connections gradually, "
            "and show how your understanding evolves as you process the information. Use natural language phrases "
            "like 'Hmm...', 'This is interesting because...', 'Wait, let me think about...', 'Actually...', "
            "'Now that I look at it...', 'This reminds me of...', 'I wonder if...', etc."
            "\n\n"

            "### COMPARISON STRUCTURE REQUIREMENTS\n\n"
            "After your comprehensive thinking process, structure your technical comparison to include:\n\n"

            "1. **Overview of Comparison**\n"
            "   - Brief context about the entities being compared\n"
            "   - Why this comparison matters from a technical perspective\n"
            "   - Key findings at a high level\n\n"

            "2. **Systematic Feature Comparison**\n"
            "   - Direct side-by-side comparison of key technical attributes\n"
            "   - Quantitative differences when available\n"
            "   - Qualitative assessments of differences\n\n"

            "3. **Architectural Analysis**\n"
            "   - Fundamental design differences\n"
            "   - Technical trade-offs in each approach\n"
            "   - Architectural strengths and limitations\n\n"

            "4. **Performance Considerations**\n"
            "   - Efficiency differences\n"
            "   - Scalability comparisons\n"
            "   - Resource requirements\n\n"

            "5. **Use Case Suitability**\n"
            "   - Scenarios where one approach excels over others\n"
            "   - Optimal application contexts\n"
            "   - Boundary conditions\n\n"

            "6. **Technical Insights**\n"
            "   - Deeper patterns or principles revealed by the comparison\n"
            "   - Technical lessons that can be applied elsewhere\n"
            "   - Unique engineering considerations\n\n"

            "7. **Information Gaps & Uncertainties**\n"
            "   - Explicit acknowledgment of missing critical information\n"
            "   - Areas where further investigation would be valuable\n\n"

            "Adapt this structure as appropriate to the specific query and available information, "
            "expanding sections with substantial data and condensing or omitting those without sufficient support."
            "\n\n"

            "### STYLE AND FORMAT GUIDANCE\n\n"
            "Present your comparison according to these principles:\n\n"

            "**Technical Precision**\n"
            "- Define technical terms upon first use\n"
            "- Use precise, unambiguous language\n"
            "- Maintain mathematical and statistical accuracy\n"
            "- Present numerical data with appropriate units and precision\n\n"

            "**Clarity and Accessibility**\n"
            "- Structure information with clear headers and subheaders\n"
            "- Use concise paragraphs focused on single main ideas\n"
            "- Employ tables for structured parameter comparisons when appropriate\n"
            "- Include visual descriptions when it enhances understanding\n\n"

            "**Evidence-Based Analysis**\n"
            "- Ground all comparisons in the data provided\n"
            "- Differentiate between direct observations and derived insights\n"
            "- Maintain transparent reasoning chains\n"
            "- Note explicitly when making connections not stated in original sources\n\n"

            "**Balance and Fairness**\n"
            "- Present balanced analysis of strengths and limitations\n"
            "- Avoid unwarranted preference for one approach\n"
            "- Acknowledge uncertainty where appropriate\n"
            "- Consider potential biases in the underlying data\n\n"

            "Your response should read as if written by a senior ML architect with deep technical knowledge "
            "and practical implementation experience, balancing theoretical understanding with "
            "pragmatic engineering considerations."
        )

        # Build user prompt with original query and all results
        user_prompt = f"Original comparison query: {original_query}\n\n"

        for i, result in enumerate(comparison_results, 1):
            user_prompt += f"=== RESULT SET #{i} ===\n"
            user_prompt += f"Query: {result.get('query', 'N/A')}\n"

            if result.get('type') == 'error':
                user_prompt += f"ERROR: {result.get('error', 'Unknown error')}\n"
                continue

            # Use the results_text field if available, otherwise just note that results were found
            if 'results_text' in result:
                user_prompt += f"Search results:\n{result.get('results_text', 'N/A')}\n\n"
            else:
                user_prompt += f"Search results: Available but not shown in detail\n\n"

        user_prompt += (
            "Please synthesize these search results into a comprehensive comparison that addresses "
            "the original query. Follow the thinking process and structure outlined in your instructions. "
            "Focus on identifying meaningful similarities and differences between the entities being compared, "
            "and organize your analysis in a way that provides clear technical insights."
        )

        response = llm_interface.generate_structured_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.25,
            max_tokens=32000
        )

        # Process result
        if isinstance(response, dict) and 'content' in response:
            content = response['content']
            content = remove_thinking_sections(content)
        else:
            content = str(response)
            content = remove_thinking_sections(content)

        return content

    def execute_command(self, command: str) -> Dict[str, Any]:
        """
        Execute command and return results

        This method encapsulates the command handling functionality from the original CLI,
        but returns results as a callable function.

        Args:
            command: Command to execute

        Returns:
            Dict: Dictionary containing command execution results
        """
        # Implementation needs to be based on actual command handling logic
        # Currently this is a simple framework
        try:
            self._log(f"Executing command: {command}")

            # Simple command processing
            if command.lower() == "exit" or command.lower() == "quit":
                return {"type": "command", "command": command, "result": "exit"}

            if command.lower() == "help":
                return {
                    "type": "command",
                    "command": command,
                    "result": {
                        "available_commands": [
                            "help - Display help information",
                            "exit/quit - Exit system",
                            # Other commands
                        ]
                    }
                }

            # If it's a query, call query processing
            if not command.startswith("/"):
                # Async query needs special handling
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(self.process_query(command))

            # Other command handling
            return {
                "type": "command",
                "command": command,
                "result": f"Unknown command: {command}"
            }

        except Exception as e:
            self._log(f"Command execution failed: {str(e)}", level="error")
            self._handle_error(e)

            return {
                "type": "error",
                "command": command,
                "error": str(e)
            }

    # The following are private methods migrated from the QueryProcessor class

    def _process_search_results(self, search_results, reranker, parsed_query, query_text,
                                max_to_return=10, rerank_threshold=0.1):
        """Process and rerank search results"""
        if not isinstance(search_results, dict) or 'items' not in search_results:
            return []

        # Extract items from search results
        items_to_rerank = search_results['items']

        # Loop through each item and add content field
        for item in items_to_rerank:
            item['content'] = ("Model description: " + item.get('merged_description', '') +
                               ", created year: " + item.get('created_year', '') +
                               ", created month: " + item.get('created_month', '') +
                               ", last modified year: " + item.get('last_modified_year', '') +
                               ", last modified month: " + item.get('last_modified_month', ''))

        if reranker and items_to_rerank:
            self._log(f"Sending {len(items_to_rerank)} items to reranker")
            return reranker.rerank(
                query=parsed_query.get("processed_query", query_text),
                results=items_to_rerank,
                top_k=max_to_return,
                threshold=rerank_threshold
            )
        else:
            return items_to_rerank

    def _remove_field(self, dict_list, field_to_remove):
        """Remove a field from all dictionaries in a list"""
        if not dict_list:
            return []
        return [{k: v for k, v in item.items() if k != field_to_remove} for item in dict_list]

    def _build_results_text(self, reranked_results):
        """Build structured text of search results"""
        import json

        results_text = ""
        for idx, model in enumerate(reranked_results, 1):
            # Ensure model dict and safely parse nested metadata
            if not isinstance(model, dict):
                model = {"model_id": str(model), "metadata": {}}

            # Extract description
            description = model.get('merged_description') or model.get('description')
            md = model.get('metadata') or {}
            if not description:
                description = md.get('description') or md.get('merged_description') or 'N/A'

            # Safely parse nested JSON fields in metadata
            for key in ["file", "framework", "architecture", "dataset", "training_config"]:
                val = md.get(key)
                if isinstance(val, str):
                    try:
                        md[key] = json.loads(val)
                    except json.JSONDecodeError:
                        md[key] = {}

            # Extract fields with fallbacks
            model_id = model.get('model_id') or model.get('id') or 'missing'
            file_md = md.get('file', {})
            fw = md.get('framework', {})
            arch = md.get('architecture', {})
            ds = md.get('dataset', {})
            training = md.get('training_config', {})

            # Compose block
            results_text += f"Model #{idx}:\n"
            results_text += f"- Model ID: {model_id}\n"
            results_text += f"- File Size: {file_md.get('size_bytes', 'missing')}\n"
            results_text += f"- Created On: {file_md.get('creation_date', 'missing')}\n"
            results_text += f"- Last Modified: {file_md.get('last_modified_date', 'missing')}\n"
            results_text += f"- Framework: {fw.get('name', 'missing')} {fw.get('version', '')}\n"
            results_text += f"- Architecture: {arch.get('type', 'missing')}\n\n{arch.get('reason', 'missing')}\n"
            results_text += f"- Dataset: {ds.get('name', 'missing')}\n"
            if training:
                results_text += "- Training Configuration:\n"
                for field in ['batch_size', 'learning_rate', 'optimizer', 'epochs', 'hardware_used']:
                    results_text += f"  - {field.replace('_', ' ').title()}: {training.get(field, 'missing')}\n"
            results_text += f"- Description: {description}\n\n"

        return results_text

    def _build_meta_prompt(self, query_text, reranked_results, llm_interface):
        """Build meta prompt for the LLM"""
        result_schema = "Model ID, File Size, Created On, Last Modified, Framework, Architecture, Dataset, Training Configuration, Description"

        # Construct prompt builder logic
        system_prompt = (
            "### META-PROMPT GENERATOR ROLE AND PURPOSE\n\n"
            "You are a senior machine-learning architect with expertise in creating clear technical documentation. Your task is to craft concise yet comprehensive meta-prompts that will generate high-quality ML system reports for engineers to learn, understand, reproduce, and extend machine learning systems.\n\n"

            "### THINKING PROCESS REQUIREMENTS\n\n"
            "Before constructing each meta-prompt, engage in thorough analytical reasoning enclosed in <thinking></thinking> tags that demonstrates:\n"
            "- Analysis of the user's specific request and technical context\n"
            "- Identification of key information requirements for the target ML engineers\n"
            "- Consideration of potential knowledge gaps that need addressing\n"
            "- Assessment of required technical depth and breadth\n"
            "- Strategy for ensuring factual accuracy while providing meaningful insights\n\n"

            "### META-PROMPT STRUCTURE REQUIREMENTS\n\n"
            "Construct a single, high-level meta-prompt that includes the following elements in a logical flow:\n\n"

            "1. **Request Contextualization**\n"
            "   - Clearly restate the user's original request to establish focus\n"
            "   - Frame the specific ML engineering context the response should address\n\n"

            "2. **Data Source Guidelines**\n"
            "   - Explicitly instruct to use ONLY the provided search results and runtime data\n"
            "   - Prohibit any form of hallucination, fabrication, or invented details\n"
            "   - Require transparency when information is incomplete or unavailable\n\n"

            "3. **Analytical Framework**\n"
            "   - Direct the LLM to explain how the search results specifically answer the user query\n"
            "   - Require analysis to be derived solely from the provided data\n"
            "   - Instruct to synthesize insights across multiple results when applicable\n\n"

            "4. **Information Awareness**\n"
            "   - Inform that there are {result_count} search results available\n"
            "   - Note that results are sorted by similarity in decreasing order\n"
            "   - Instruct to prioritize higher-ranked results while considering all relevant information\n\n"

            "5. **Technical Communication Standards**\n"
            "   - Establish the role as a senior ML architect writing for technical ML engineers\n"
            "   - Require clear definitions for all technical terms without exception\n"
            "   - Mandate systematic explanation that leaves no conceptual gaps\n"
            "   - Instruct to use appropriate technical precision while remaining accessible\n\n"

            "6. **Response Quality Requirements**\n"
            "   - Require meaningful, substantive responses rather than restatements or generalities\n"
            "   - Mandate explicit \"No data found\" statements when search results lack relevant information\n"
            "   - Instruct to highlight limitations in the available data when present\n\n"

            "### STYLE AND FORMAT GUIDANCE\n\n"
            "- The meta-prompt should be concise yet complete (typically 150-250 words)\n"
            "- Avoid any references to other LLM identifiers or systems\n"
            "- Focus entirely on the user's intent and technical requirements\n"
            "- Use precise technical language appropriate for ML engineering contexts\n"
            "- Structure instructions in a natural, logical progression\n\n"

            "### REMINDERS\n\n"
            "- DO NOT answer the user's query yourself - return only the meta-prompt text\n"
            "- Include NO actual data in your response - the meta-prompt is a template for future use\n"
            "- Ensure your thinking process demonstrates genuine technical reasoning rather than mechanical steps\n"
            "- The ideal meta-prompt balances technical precision with clarity and completeness"
        )

        user_prompt = (
            f"User query: {query_text}\n"
            f"Result schema: {result_schema}\n"
            f"Number of results: {len(reranked_results)}\n"
            "DO NOT answer the user's query yourself - return only the meta-prompt text\n"
        )

        builder_response = llm_interface.generate_structured_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.05,
            max_tokens=32000
        )

        # Safely extract constructed prompt text
        if isinstance(builder_response, dict):
            constructed_prompt = (
                    builder_response.get('content')
                    or builder_response.get('text')
                    or (builder_response.get('message') or {}).get('content')
                    or str(builder_response))
        else:
            constructed_prompt = str(builder_response)

        return remove_thinking_sections(constructed_prompt.strip()).strip()

    # Internal helper methods

    def _log(self, message: str, level: str = "info") -> None:
        """Internal logging method"""
        if level == "error":
            logging.error(message)
        elif level == "warning":
            logging.warning(message)
        else:
            logging.info(message)

        # Call log callback
        self.callbacks["on_log"]({"level": level, "message": message})

    def _update_status(self, status: str) -> None:
        """Update system status"""
        self.callbacks["on_status"](status)

    def _handle_result(self, result: Dict[str, Any]) -> None:
        """Handle result"""
        self.callbacks["on_result"](result)

    def _handle_error(self, error: Exception) -> None:
        """Handle error"""
        self.callbacks["on_error"](error)