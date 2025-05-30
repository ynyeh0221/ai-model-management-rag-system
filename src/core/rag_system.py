"""
RAG System Core Module - Workflow Documentation

┌─────────────────────────────────────────────────────────────────────────────────┐
│                           RAG SYSTEM ARCHITECTURE OVERVIEW                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────┐    ┌──────────────────┐    ┌─────────────────────────────────┐ │
│  │   User UI   │───▶│   RAGSystem      │───▶│       Components                │ │
│  │ (CLI/Web)   │    │   Core Class     │    │                                 │ │
│  └─────────────┘    └──────────────────┘    │ ┌─────────────────────────────┐ │ │
│                              │              │ │     Query Engine            │ │ │
│                              │              │ │ • Parser                    │ │ │
│                              │              │ │ • Search Dispatcher         │ │ │
│                              │              │ │ • Analytics                 │ │ │
│                              │              │ │ • Reranker                  │ │ │
│                              │              │ └─────────────────────────────┘ │ │
│                              │              │                                 │ │
│                              │              │ ┌─────────────────────────────┐ │ │
│                              │              │ │   Response Generator        │ │ │
│                              │              │ │ • LLM Interface             │ │ │
│                              │              │ └─────────────────────────────┘ │ │
│                              │              └─────────────────────────────────┘ │
│                              │                                                  │
│                              ▼                                                  │
│                    ┌──────────────────┐                                         │
│                    │    Callbacks     │                                         │
│                    │ • on_log         │                                         │
│                    │ • on_result      │                                         │
│                    │ • on_error       │                                         │
│                    │ • on_status      │                                         │
│                    └──────────────────┘                                         │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                          MAIN QUERY PROCESSING WORKFLOW                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  User Query                                                                     │
│      │                                                                          │
│      ▼                                                                          │
│  ┌─────────────────┐                                                            │
│  │ process_query() │                                                            │
│  └─────────────────┘                                                            │
│          │                                                                      │
│          ▼                                                                      │
│  ┌─────────────────┐     NO    ┌──────────────────────────────────────────┐     │
│  │ Clarity Check   │──────────▶│           Skip Clarity Check             │     │
│  │ Enabled?        │           └──────────────────────────────────────────┘     │
│  └─────────────────┘                                    │                       │
│          │ YES                                          │                       │
│          ▼                                              │                       │
│  ┌─────────────────┐                                    │                       │
│  │ Check Query     │                                    │                       │
│  │ Clarity         │                                    │                       │
│  └─────────────────┘                                    │                       │
│          │                                              │                       │
│          ▼                                              │                       │
│  ┌─────────────────┐   UNCLEAR  ┌──────────────────┐    │                       │
│  │ Query Clear?    │───────────▶│ Return           │    │                       │
│  │                 │            │ Clarification    │    │                       │
│  └─────────────────┘            │ Request          │    │                       │
│          │ CLEAR                └──────────────────┘    │                       │
│          ▼                                              │                       │
│  ┌─────────────────┐                                    │                       │
│  │ Use Improved    │                                    │                       │
│  │ Query if        │                                    │                       │
│  │ Available       │                                    │                       │
│  └─────────────────┘                                    │                       │
│          │                                              │                       │
│          ▼                                              ▼                       │
│  ┌─────────────────┐     NO    ┌──────────────────────────────────────────┐     │
│  │ Comparison      │──────────▶│        Skip Comparison Detection         │     │
│  │ Detection       │           └──────────────────────────────────────────┘     │
│  │ Enabled?        │                                    │                       │
│  └─────────────────┘                                    │                       │
│          │ YES                                          │                       │
│          ▼                                              │                       │
│  ┌─────────────────┐                                    │                       │
│  │ Detect          │                                    │                       │
│  │ Comparison      │                                    │                       │
│  │ Query           │                                    │                       │
│  └─────────────────┘                                    │                       │
│          │                                              │                       │
│          ▼                                              │                       │
│  ┌─────────────────┐  YES     ┌──────────────────┐      │                       │
│  │ Is Comparison   │─────────▶│ Process          │      │                       │
│  │ Query?          │          │ Comparison       │      │                       │
│  └─────────────────┘          │ Query            │      │                       │
│          │ NO                 └──────────────────┘      │                       │
│          │                              │               │                       │
│          │                              ▼               ▼                       │
│          │                    ┌──────────────────────────────────────────┐      │
│          └───────────────────▶│         Process Regular Query            │      │
│                               └──────────────────────────────────────────┘      │
│                                              │                                  │
│                                              ▼                                  │
│                               ┌──────────────────────────────────────────┐      │
│                               │           Return Results                 │      │
│                               └──────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                        COMPARISON QUERY WORKFLOW                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Comparison Query                                                               │
│      │                                                                          │
│      ▼                                                                          │
│  ┌─────────────────────────────────────┐                                        │
│  │ _process_comparison_query()         │                                        │
│  └─────────────────────────────────────┘                                        │
│                      │                                                          │
│                      ▼                                                          │
│  ┌─────────────────────────────────────┐                                        │
│  │ For each retrieval query:           │                                        │
│  │ ┌─────────────────────────────────┐ │                                        │
│  │ │ Sub-Query 1: "Entity A info"    │ │                                        │
│  │ └─────────────────────────────────┘ │                                        │
│  │ ┌─────────────────────────────────┐ │                                        │
│  │ │ Sub-Query 2: "Entity B info"    │ │                                        │
│  │ └─────────────────────────────────┘ │                                        │
│  │ ┌─────────────────────────────────┐ │                                        │
│  │ │ Sub-Query N: "Entity N info"    │ │                                        │
│  │ └─────────────────────────────────┘ │                                        │
│  └─────────────────────────────────────┘                                        │
│                      │                                                          │
│                      ▼                                                          │
│  ┌─────────────────────────────────────┐                                        │
│  │ Process each sub-query:             │                                        │
│  │                                     │                                        │
│  │ Sub-Query ──▶ Parse ──▶ Search ──▶  │                                        │
│  │                │          │         │                                        │
│  │                ▼          ▼         │                                        │
│  │            Analytics   Results      │                                        │
│  │                │          │         │                                        │
│  │                ▼          ▼         │                                        │
│  │           Log Query   Rerank        │                                        │
│  │                          │          │                                        │
│  │                          ▼          │                                        │
│  │                   Store Results     │                                        │
│  │                                     │                                        │
│  │ (Skip LLM response generation)      │                                        │
│  └─────────────────────────────────────┘                                        │
│                      │                                                          │
│                      ▼                                                          │
│  ┌─────────────────────────────────────┐                                        │
│  │ _generate_comparison_response()     │                                        │
│  │                                     │                                        │
│  │ Input: Original query +             │                                        │
│  │        All sub-query results        │                                        │
│  │                                     │                                        │
│  │ ┌─────────────────────────────────┐ │                                        │
│  │ │         LLM Synthesis           │ │                                        │
│  │ │                                 │ │                                        │
│  │ │ • Analyze all results           │ │                                        │
│  │ │ • Identify similarities         │ │                                        │
│  │ │ • Highlight differences         │ │                                        │
│  │ │ • Generate comparison table     │ │                                        │
│  │ │ • Provide recommendations       │ │                                        │
│  │ └─────────────────────────────────┘ │                                        │
│  └─────────────────────────────────────┘                                        │
│                      │                                                          │
│                      ▼                                                          │
│  ┌─────────────────────────────────────┐                                        │
│  │ Return Comparison Result:           │                                        │
│  │ • Type: "comparison_search"         │                                        │
│  │ • Original query                    │                                        │
│  │ • Sub-queries list                  │                                        │
│  │ • Individual results                │                                        │
│  │ • Final comparison response         │                                        │
│  └─────────────────────────────────────┘                                        │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                         REGULAR QUERY WORKFLOW                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Regular Query                                                                  │
│      │                                                                          │
│      ▼                                                                          │
│  ┌─────────────────────────────────────┐                                        │
│  │ _process_regular_query()            │                                        │
│  └─────────────────────────────────────┘                                        │
│                      │                                                          │
│                      ▼                                                          │
│  ┌─────────────────────────────────────┐                                        │
│  │ Parse Query                         │                                        │
│  │ ┌─────────────────────────────────┐ │                                        │
│  │ │ • Extract intent                │ │                                        │
│  │ │ • Process parameters            │ │                                        │
│  │ │ • Clean query text              │ │                                        │
│  │ └─────────────────────────────────┘ │                                        │
│  └─────────────────────────────────────┘                                        │
│                      │                                                          │
│                      ▼                                                          │
│  ┌─────────────────────────────────────┐                                        │
│  │ Log Query Analytics                 │                                        │
│  └─────────────────────────────────────┘                                        │
│                      │                                                          │
│                      ▼                                                          │
│  ┌─────────────────────────────────────┐                                        │
│  │ Dispatch Search                     │                                        │
│  │ ┌─────────────────────────────────┐ │                                        │
│  │ │ Input: processed_query, intent, │ │                                        │
│  │ │        parameters, user_id      │ │                                        │
│  │ │                                 │ │                                        │
│  │ │ Output: search_results          │ │                                        │
│  │ └─────────────────────────────────┘ │                                        │
│  └─────────────────────────────────────┘                                        │
│                      │                                                          │
│                      ▼                                                          │
│  ┌─────────────────────────────────────┐                                        │
│  │ Intent Check                        │                                        │
│  └─────────────────────────────────────┘                                        │
│          │                     │                                                │
│          ▼                     ▼                                                │
│  ┌──────────────┐    ┌──────────────────────┐                                   │
│  │ Image Search │    │    Text Search       │                                   │
│  │              │    │                      │                                   │
│  │ Return:      │    │ ┌──────────────────┐ │                                   │
│  │ • Type:      │    │ │ Process & Rerank │ │                                   │
│  │   "image_    │    │ │ Search Results   │ │                                   │
│  │   search"    │    │ └──────────────────┘ │                                   │
│  │ • Results    │    │          │           │                                   │
│  │ • Query      │    │          ▼           │                                   │
│  │ • Parsed     │    │ ┌──────────────────┐ │                                   │
│  │   query      │    │ │ Paginated        │ │                                   │
│  └──────────────┘    │ │ Processing:      │ │                                   │
│                      │ │                  │ │                                   │
│                      │ │ Page 1 (3 items) │ │                                   │
│                      │ │ ┌──────────────┐ │ │                                   │
│                      │ │ │ Build Results│ │ │                                   │
│                      │ │ │ Text         │ │ │                                   │
│                      │ │ └──────────────┘ │ │                                   │
│                      │ │        │         │ │                                   │
│                      │ │        ▼         │ │                                   │
│                      │ │ ┌──────────────┐ │ │                                   │
│                      │ │ │ Generate LLM │ │ │                                   │
│                      │ │ │ Response     │ │ │                                   │
│                      │ │ └──────────────┘ │ │                                   │
│                      │ │        │         │ │                                   │
│                      │ │        ▼         │ │                                   │
│                      │ │ ┌──────────────┐ │ │                                   │
│                      │ │ │ Need More    │ │ │                                   │
│                      │ │ │ Results?     │ │ │                                   │
│                      │ │ └──────────────┘ │ │                                   │
│                      │ │    │         │   │ │                                   │
│                      │ │    ▼YES      │NO │ │                                   │
│                      │ │ Page 2,3...  │   │ │                                   │
│                      │ │              ▼   │ │                                   │
│                      │ │            Done  │ │                                   │
│                      │ └──────────────────┘ │                                   │
│                      │                      │                                   │
│                      │ Return:              │                                   │
│                      │ • Type: "text_       │                                   │
│                      │   search"            │                                   │
│                      │ • Query              │                                   │
│                      │ • Parsed query       │                                   │
│                      │ • Search results     │                                   │
│                      │ • Pages checked      │                                   │
│                      │ • Final response     │                                   │
│                      └──────────────────────┘                                   │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                           COMPONENT INTERACTIONS                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Components Dictionary Structure:                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ {                                                                       │    │
│  │   "query_engine": {                                                     │    │
│  │     "query_parser": QueryParser,                                        │    │
│  │     "search_dispatcher": SearchDispatcher,                              │    │
│  │     "query_analytics": QueryAnalytics,                                  │    │
│  │     "reranker": Reranker                                                │    │
│  │   },                                                                    │    │
│  │   "response_generator": {                                               │    │
│  │     "llm_interface": LLMInterface                                       │    │
│  │   }                                                                     │    │
│  │ }                                                                       │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                 │
│  Query Processing Flow:                                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌──────────────────┐                     │
│  │ User Query  │───▶│ QueryParser │───▶│ SearchDispatcher │                     │
│  └─────────────┘    └─────────────┘    └──────────────────┘                     │
│                              │                   │                              │
│                              ▼                   ▼                              │
│                    ┌─────────────────┐ ┌─────────────────┐                      │
│                    │ QueryAnalytics  │ │ Search Results  │                      │
│                    │ (Log Query)     │ └─────────────────┘                      │
│                    └─────────────────┘           │                              │
│                                                  ▼                              │
│                                        ┌─────────────────┐                      │
│                                        │    Reranker     │                      │
│                                        │ (Score & Sort)  │                      │
│                                        └─────────────────┘                      │
│                                                  │                              │
│                                                  ▼                              │
│                                        ┌─────────────────┐                      │
│                                        │  LLMInterface   │                      │
│                                        │ (Generate       │                      │
│                                        │  Response)      │                      │
│                                        └─────────────────┘                      │
│                                                                                 │
│  Callback Flow:                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐              │
│  │ _log()          │───▶│ on_log callback │───▶│ UI Layer        │              │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘              │
│                                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐              │
│  │ _update_status()│───▶│ on_status       │───▶│ Progress        │              │
│  └─────────────────┘    │ callback        │    │ Indicator       │              │
│                         └─────────────────┘    └─────────────────┘              │
│                                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐              │
│  │ _handle_result()│───▶│ on_result       │───▶│ Display         │              │
│  └─────────────────┘    │ callback        │    │ Results         │              │
│                         └─────────────────┘    └─────────────────┘              │
│                                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐              │
│  │ _handle_error() │───▶│ on_error        │───▶│ Error           │              │
│  └─────────────────┘    │ callback        │    │ Handling        │              │
│                         └─────────────────┘    └─────────────────┘              │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                            STATUS PROGRESSION                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  System Lifecycle:                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐       │
│  │ initialized │───▶│    ready    │───▶│ processing  │───▶│  completed  │       │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘       │
│                                               │                                 │
│                                               ▼                                 │
│                                        ┌─────────────┐                          │
│                                        │    error    │                          │
│                                        └─────────────┘                          │
│                                                                                 │
│  Query Processing Status Flow:                                                  │
│  processing ──▶ searching ──▶ processing_results ──▶ generating_response        │
│       │              │               │                        │                 │
│       ▼              ▼               ▼                        ▼                 │
│  needs_clarification │         processing_comparison      completed             │
│                      ▼               │                        │                 │
│              generating_comparison   ▼                        │                 │
│                      │          generating_response           │                 │
│                      │               │                        │                 │
│                      ▼               ▼                        ▼                 │
│                  completed       completed               error (if failed)      │
└─────────────────────────────────────────────────────────────────────────────────┘
"""

import asyncio
import logging
from typing import Dict, Any, Callable, List, Tuple

from src.core.prompt_manager.query_path_prompt_manager import QueryPathPromptManager


class RAGSystem:
    """
    RAG System Core Class - Encapsulating all core functionality into callable functions

    This class provides all core functionality of the RAG system, including:
    - Initializing system components
    - Processing user queries
    - Generating responses
    - Handling model and image display

    By encapsulating these functionalities in a single class, we can easily reuse them in different cli_interface (CLI, Web, etc.).
    """

    def __init__(self):
        """Initialize RAG system core components"""
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
        Initialize RAG system components

        Args:
            components: Dictionary containing initialized system components
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

    async def process_query(self, query_text: str, enable_clarity_check: bool = False, enable_comparison_detection: bool = True) -> Dict[str, Any]:
        """
        Process user query and return results

        This method encapsulates the process_query functionality from the original CLI,
        but returns results as a callable function instead of printing directly to the console.

        Args:
            query_text: User query text
            enable_comparison_detection: Whether to enable comparison detection and processing

        Returns:
            Dict: Dictionary containing query results
            :param query_text:
            :param enable_comparison_detection:
            :param enable_clarity_check:
        """
        try:
            self._log(f"Processing query: {query_text}")
            self._update_status("processing")

            # Ensure components are initialized
            if not self.components:
                raise ValueError("System components not initialized")

            # Extract required components
            llm_interface = self.components["response_generator"]["llm_interface"]

            # Check query clarity if enabled
            if enable_clarity_check:
                clarity_result = await self._check_query_clarity(query_text, llm_interface)

                # If query is not clear, return a special result for the interface to handle
                if not clarity_result['is_clear']:
                    self._log(f"Query needs clarification: {clarity_result['reason']}")
                    self._update_status("needs_clarification")

                    result = {
                        "type": "needs_clarification",
                        "query": query_text,
                        "clarity_result": clarity_result
                    }

                    self._handle_result(result)
                    return result

                # If we have an improved query, use it
                if clarity_result['improved_query'] != query_text:
                    self._log(f"Query improved from: '{query_text}' to: '{clarity_result['improved_query']}'")
                    query_text = clarity_result['improved_query']


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

    async def _check_query_clarity(self, query_text: str, llm_interface) -> Dict[str, Any]:
        """
        Check query clarity and suggest improvements if needed.

        Args:
            query_text: The original user query
            llm_interface: Interface to the LLM

        Returns:
            Dict: Dictionary containing clarity assessment
        """
        system_prompt = QueryPathPromptManager.get_system_prompt_for_query_clarity()
        user_prompt = f"Analyze this query: {query_text}"

        self._log(f"Checking clarity of query: {query_text}")

        response = llm_interface.generate_structured_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.05,
            max_tokens=10000
        )

        if isinstance(response, dict) and 'content' in response:
            content = response['content']
        else:
            content = str(response)

        import json
        import re

        # Try to extract JSON from content
        json_pattern = r'```json\s*(.*?)\s*```'
        match = re.search(json_pattern, content, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            json_pattern = r'({.*})'
            match = re.search(json_pattern, content, re.DOTALL)
            if match:
                json_str = match.group(1)
            else:
                json_str = content

        try:
            result = json.loads(json_str)
            self._log(f"Query clarity result: {result}")
            return {
                'is_clear': result.get('is_clear', True),
                'improved_query': result.get('improved_query', query_text),
                'suggestions': result.get('suggestions', []),
                'reason': result.get('reason', "")
            }
        except Exception as e:
            self._log(f"Error parsing query clarity response: {str(e)}", level="error")
            return {
                'is_clear': True,  # Default to assuming query is clear on error
                'improved_query': query_text,
                'suggestions': [],
                'reason': f"Error analyzing query clarity: {str(e)}"
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
            all_reranked = self._process_search_results(
                search_results, reranker, parsed_query, query_text, max_to_return=max_output_counts, rerank_threshold=0
            )

            pages_info = []
            page_id = 0
            def process_paged_results(all_reranked, page_id: int, page_size: int, max_pages_to_fetch: int):

                print(f"Processing page {page_id}")

                pages_info.append({})

                current_page = all_reranked[page_id * page_size:(page_id + 1) * page_size]

                # Build results text for current page
                results_text = self._build_results_text(current_page, page_id != len(all_reranked) - 1)
                print(f"results_text: {results_text}")

                # Initialize result structure
                pages_info[-1] = {
                    "search_results": current_page,
                    "has_next_page": page_id == len(all_reranked) - 1,
                }

                # Generate LLM response if requested
                if generate_llm_response:
                    # TODO Refactor this prompt to make it output response in desired format when INSUFFICIENT INFORMATION (MORE RESULTS NEEDED)
                    system_prompt = QueryPathPromptManager.get_system_prompt_for_regular_response()

                    # Build user prompt
                    user_prompt = f"\nUser query:\n{query_text}\n"
                    user_prompt += f"Search results:\n{results_text}"
                    if "page_summary" in pages_info[-1]:
                        summaries = '\n\n'.join([page_info["page_summary"] for page_info in pages_info])
                        user_prompt += f"\nSummaries of previous pages: {summaries}"

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
                        content = self.remove_thinking_sections(content)
                    else:
                        content = str(final_response)
                        content = self.remove_thinking_sections(content)

                    pages_info[-1]["page_summary"] = content

                    # If llm decides beed more results, then fetch more results
                    if "<need_more_results>" in content and max_pages_to_fetch > 0:
                        process_paged_results(all_reranked, page_id + 1, page_size, max_pages_to_fetch - 1)
                else:
                    # Skip LLM response generation
                    pages_info[-1]["page_summary"] = ""

            # TODO make page_size and max_pages_to_fetch defined and passed-in from outside
            process_paged_results(all_reranked, 0, 3, 3 if generate_llm_response else max_output_counts)

            result = {"type": "text_search" if generate_llm_response else "retrieval_only", "query": query_text,
                      "parsed_query": parsed_query,
                      "search_results": [item for page_info in pages_info for item in page_info["search_results"]],
                      "checked_pages": page_id + 1, "final_response": pages_info[-1]["page_summary"]}

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
        system_prompt = QueryPathPromptManager.get_system_prompt_for_comparison_query()
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

            # Extract JSON from the response (it might be wrapped in Markdown code blocks)
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
        system_prompt = QueryPathPromptManager.generate_system_prompt_for_comparison_response()
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
            content = self.remove_thinking_sections(content)
        else:
            content = str(response)
            content = self.remove_thinking_sections(content)

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

    def _process_search_results(self, search_results, reranker, parsed_query, query_text,
                                max_to_return=10, rerank_threshold=0.1):
        """Process and rerank search results"""
        if not isinstance(search_results, dict) or 'items' not in search_results:
            return []

        # Extract items from search results
        items_to_rerank = search_results['items']

        if reranker and items_to_rerank:
            self._log(f"Sending {len(items_to_rerank)} items to reranker")
            # Loop through each item and add content field
            for item in items_to_rerank:
                item['content'] = ("Model description: " + item.get('merged_description', '') + "\n" +
                                   "architecture is: " + item.get('metadata', {}).get('architecture', '') + "\n" +
                                   "dataset is: " + item.get('metadata', {}).get('dataset', {})
                                   )
            all_ranked = reranker.rerank(
                query=parsed_query.get("processed_query", query_text),
                results=items_to_rerank,
                top_k=len(items_to_rerank),
                threshold=rerank_threshold
            )
            all_ranked = self._remove_field(all_ranked, "content")
            return all_ranked
        else:
            return items_to_rerank

    @staticmethod
    def _remove_field(dict_list, field_to_remove):
        """Remove a field from all dictionaries in a list"""
        if not dict_list:
            return []
        return [{k: v for k, v in item.items() if k != field_to_remove} for item in dict_list]

    @staticmethod
    def _build_results_text(reranked_results, has_next_page: bool):
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
            rerank_score = model.get('rerank_score', 'N/A')
            file_md = md.get('file', {})
            fw = md.get('framework', {})
            arch = md.get('architecture', {})
            ds = md.get('dataset', {})
            training = md.get('training_config', {})

            # Compose block
            results_text += f"Model #{idx}:\n"
            results_text += f"- Model ID: {model_id}\n"
            results_text += f"- Rerank Score: {rerank_score}\n"
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

            results_text += f"Has More Models: {has_next_page}\n"

        return results_text

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

    @staticmethod
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
