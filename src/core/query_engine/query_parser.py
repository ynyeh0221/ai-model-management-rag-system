"""
QueryParser Workflow ASCII Diagram

┌─────────────────────────────────────────────────────────────────────────────────┐
│                              QUERYPARSER INITIALIZATION                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │   spaCy     │  │    NLTK     │  │ LangChain   │  │     Regex Patterns      │ │
│  │   Model     │  │ Components  │  │   LLM       │  │  (model_id, metrics,    │ │
│  │(NLP Parser) │  │(Lemmatizer, │  │ (Ollama)    │  │   filters, limits)      │ │
│  │             │  │ Stopwords)  │  │             │  │                         │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                PARSE_QUERY()                                    │
│                            [Main Entry Point]                                   │
└─────────────────────────────────────────────────────────────────────────────────┘
                                         │
                         ┌───────────────┼───────────────┐
                         ▼               ▼               ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────────────────────────┐
│ PREPROCESS_     │ │ CLASSIFY_       │ │        EXTRACT_PARAMETERS()         │
│ QUERY()         │ │ INTENT()        │ │                                     │
├─────────────────┤ ├─────────────────┤ ├─────────────────────────────────────┤
│ 1. Text cleaning│ │ 1. LangChain    │ │ 1. Basic regex extraction           │
│ 2. spaCy parse  │ │    system       │ │ 2. Model ID detection               │
│ 3. Preserve     │ │    prompt       │ │ 3. LLM-based NER                    │
│    entities     │ │ 2. LLM          │ │ 4. Intent-specific params           │
│ 4. Preserve     │ │    inference    │ │                                     │
│    noun phrases │ │ 3. JSON         │ │ ┌─────────────────────────────────┐ │
│ 5. Lemmatize    │ │    parsing      │ │ │      Sub-extractors:            │ │
│    remaining    │ │ 4. Return       │ │ │  • _extract_model_id_mentions() │ │
│    tokens       │ │    QueryIntent  │ │ │  • _extract_entities_with_llm() │ │
└─────────────────┘ └─────────────────┘ │ │  • _extract_image_parameters()  │ │
         │                   │          │ │    (if IMAGE_SEARCH intent)     │ │
         ▼                   ▼          │ └─────────────────────────────────┘ │
┌─────────────────┐ ┌─────────────────┐ └─────────────────────────────────────┘
│ Cleaned Query   │ │ Intent + Reason │                   │
│ Text            │ │                 │                   ▼
└─────────────────┘ └─────────────────┘ ┌─────────────────────────────────────┐
                                        │        Parameter Extraction         │
                                        │             Sub-flows               │
                                        ├─────────────────────────────────────┤
                                        │                                     │
                                        │ ┌─────────────────────────────────┐ │
                                        │ │     REGEX-BASED EXTRACTION      │ │
                                        │ │  • Model IDs (model_id_pattern) │ │
                                        │ │  • Metrics (accuracy, loss...)  │ │
                                        │ │  • Filters (architecture...)    │ │
                                        │ │  • Limits (top 5, first 10...)  │ │
                                        │ │  • Sort parameters              │ │
                                        │ │  • Date/year patterns           │ │
                                        │ └─────────────────────────────────┘ │
                                        │                │                    │
                                        │                ▼                    │
                                        │ ┌─────────────────────────────────┐ │
                                        │ │       LLM-BASED NER             │ │
                                        │ │  (_extract_entities_with_llm)   │ │
                                        │ │                                 │ │
                                        │ │  Input: Query text              │ │
                                        │ │  ┌─────────────────────────────┐│ │
                                        │ │  │   LangChain Chat Pipeline   ││ │
                                        │ │  │  ┌─────────────────────────┐││ │
                                        │ │  │  │    System Message       │││ │
                                        │ │  │  │  (NER instructions)     │││ │
                                        │ │  │  └─────────────────────────┘││ │
                                        │ │  │  ┌─────────────────────────┐││ │
                                        │ │  │  │    Human Message        │││ │
                                        │ │  │  │   (User query)          │││ │
                                        │ │  │  └─────────────────────────┘││ │
                                        │ │  │  ┌─────────────────────────┐││ │
                                        │ │  │  │       LLM               │││ │
                                        │ │  │  │   (Ollama model)        │││ │
                                        │ │  │  └─────────────────────────┘││ │
                                        │ │  └─────────────────────────────┘│ │
                                        │ │  │                              │ │
                                        │ │  ▼                              │ │
                                        │ │  Output: Structured JSON        │ │
                                        │ │  {                              │ │
                                        │ │    "architecture": {...},       │ │
                                        │ │    "dataset": {...},            │ │
                                        │ │    "training_config": {...}     │ │
                                        │ │  }                              │ │
                                        │ └─────────────────────────────────┘ │
                                        │                │                    │
                                        │                ▼                    │
                                        │ ┌─────────────────────────────────┐ │
                                        │ │   INTENT-SPECIFIC EXTRACTION    │ │
                                        │ │                                 │ │
                                        │ │  If intent == IMAGE_SEARCH:     │ │
                                        │ │  ┌─────────────────────────────┐│ │
                                        │ │  │ _extract_image_parameters() ││ │
                                        │ │  │  • Search type detection    ││ │
                                        │ │  │  • Epoch/tag/color filters  ││ │
                                        │ │  │  • Content/style filters    ││ │
                                        │ │  │  • Resolution preferences   ││ │
                                        │ │  └─────────────────────────────┘│ │
                                        │ └─────────────────────────────────┘ │
                                        └─────────────────────────────────────┘
                                                          │
                                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              FINAL OUTPUT                                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│  {                                                                              │
│    "intent": "retrieval|image_search|...",                                      │
│    "reason": "Explanation from LLM",                                            │
│    "type": "retrieval|image_search|...",  // backward compatibility             │
│    "parameters": {                                                              │
│      "filters": {                        // Regex-extracted filters             │
│        "model_id": "...",                                                       │
│        "created_year": "2023",                                                  │
│        "created_month": "January"                                               │
│      },                                                                         │
│      "ner_filters": {                    // LLM-extracted entities              │
│        "architecture": {"value": "transformer", "is_positive": true},           │
│        "dataset": {"value": "ImageNet", "is_positive": true},                   │
│        "training_config": {                                                     │
│          "batch_size": {"value": 32, "is_positive": true},                      │
│          "learning_rate": {"value": 0.001, "is_positive": true},                │
│          "optimizer": {"value": "Adam", "is_positive": true},                   │
│          "epochs": {"value": 100, "is_positive": true},                         │
│          "hardware_used": {"value": "GPU", "is_positive": true}                 │
│        }                                                                        │
│      },                                                                         │
│      "limit": 10,                       // Result limit                         │
│      "sort_by": {"field": "...", "order": "..."},                               │
│      // Image search specific parameters (if applicable)                        │
│      "search_type": "similarity|epoch|tag|color|date|content|model_id",         │
│      "epoch": 42,                                                               │
│      "tags": ["tag1", "tag2"],                                                  │
│      "colors": ["red", "blue"],                                                 │
│      "prompt_terms": "...",                                                     │
│      "style_tags": ["abstract", "modern"]                                       │
│    },                                                                           │
│    "processed_query": "cleaned and lemmatized query text"                       │
│  }                                                                              │
└─────────────────────────────────────────────────────────────────────────────────┘

Key Components:
┌─────────────────────────────────────────────────────────────────────────────────┐
│ • spaCy: Named entity recognition, noun phrase extraction, POS tagging          │
│ • NLTK: Lemmatization, stopword filtering                                       │
│ • LangChain + Ollama: Intent classification and NER via LLM                     │
│ • Regex Patterns: Rule-based parameter extraction for known formats             │
│ • QueryPathPromptManager: Provides system prompts for LLM operations            │
└─────────────────────────────────────────────────────────────────────────────────┘

Flow Summary:
1. Raw query → Preprocessing (clean, normalize, preserve entities)
2. Processed query → Intent classification (LLM-based with structured JSON output)
3. Query + Intent → Parameter extraction (hybrid regex + LLM approach)
4. All components → Structured result dictionary with intent and extracted parameters
"""
import json
import logging
import re
from typing import Dict, List, Any, Optional, Union, Set

import nltk
import spacy
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_ollama import OllamaLLM
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from src.core.prompt_manager.query_path_prompt_manager import QueryPathPromptManager
from src.core.query_engine.query_intent import QueryIntent


class QueryParser:
    """
    Parser for natural language queries related to AI models.

    This class is responsible for parsing natural language queries into structured
    representations, including intent classification and parameter extraction.
    It combines both rule-based pattern matching and LLM-based intent classification
    to effectively handle a wide range of query formats.

    Attributes:
        logger: A logging instance for tracking operations and errors.
        nlp: A spaCy language model for NLP processing.
        lemmatizer: A WordNet lemmatizer for word normalization.
        stop_words: A set of common stopwords to filter out.
        model_id_pattern: Regex pattern for extracting model IDs.
        metric_pattern: Regex pattern for extracting metrics.
        filter_patterns: Dictionary of regex patterns for different filter types.
        limit_pattern: Regex pattern for extracting result limits.
        sort_pattern: Regex pattern for extracting sort parameters.
        model_families: List of common model family names for NLP processing.
        created_year_patterns: List of regex patterns for extracting creation years.

    Example:
        >>> parser = QueryParser(nlp_model="en_core_web_sm", use_langchain=True)
        >>> result = parser.parse_query("Find models with architecture transformer limit 5")
        >>> print(result["intent"]) # 'retrieval'
        >>> print(result["parameters"]["limit"]) # 5
    """

    def __init__(self, nlp_model: str = "en_core_web_sm", llm_model_name: str = "deepseek-r1:7b"):
        """
        Initialize the QueryParser with necessary NLP parts.

        Args:
            nlp_model: The spaCy model to use for NLP tasks
            llm_model_name: The name of the language model to use
        """
        # Set up logging
        self.logger = logging.getLogger(__name__)

        # Initialize NLP components
        try:
            self.nlp = spacy.load(nlp_model)
            self.logger.info(f"Loaded spaCy model: {nlp_model}")
        except OSError:
            self.logger.warning(f"Could not load spaCy model: {nlp_model}. Running spacy download...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", nlp_model], check=True)
            self.nlp = spacy.load(nlp_model)

        # Ensure NLTK resources are available
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            self.logger.info("Downloading required NLTK resources...")
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')

        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

        self.langchain_llm = OllamaLLM(model=llm_model_name, temperature=0, num_predict=10000)

        # Initialize pattern dictionaries for rule-based parsing
        self._init_patterns()

    def _init_patterns(self):
        """Initialize regex patterns and keywords for rule-based parsing."""

        # Parameter extraction patterns

        """
        Regular expression pattern to extract model IDs from text.

        Pattern: r"\b(?:model[-\s_]*id|modelid)(?:\s*[:=]\s*|\s+)([^\s]+)(?:\s|$)"

        This pattern can handle the following cases:
        - model id my_model
        - model id=my_model
        - model id = my_model
        - model id: my_model
        - model id : my_model
        - model-id my_model
        - model-id=my_model
        - model-id = my_model
        - model-id: my_model
        - model-id : my_model
        - model_id my_model
        - model_id=my_model
        - model_id = my_model
        - model_id: my_model
        - model_id : my_model
        - modelid my_model
        - modelid=my_model
        - modelid = my_model
        - modelid: my_model
        - modelid : my_model

        The pattern components:
        1. \b(?:model[-\s_]*id|modelid) - Matches different forms of "model id"
           - model id, model-id, model_id, model--id, model  id, etc.
           - Also matches modelid as a special case
        2. (?:\s*[:=]\s*|\s+) - Matches the delimiter between "model id" and the value
           - Can be a colon or equals with optional surrounding spaces
           - Or just spaces without a delimiter
        3. ([^\s]+) - Captures the model ID (any non-whitespace characters)
        4. (?:\s|$) - Ensures the model ID is followed by whitespace or end of string

        Returns a match object where group(1) contains the extracted model ID.
        """
        self.model_id_pattern = r"\b(?:model[-\s_]*id|modelid)(?:\s*[:=]\s*|\s+)([^\s]+)(?:\s|$)"
        self.metric_pattern = r"(accuracy|loss|perplexity|clip[_-]?score|performance|precision|recall|f1|mae|mse|rmse)"

        self.filter_patterns = {
            "architecture": r"architecture[:\s]+(transformer|cnn|rnn|mlp|diffusion|gan)",
            "framework": r"framework[:\s]+(pytorch|tensorflow|jax)",
            "params": r"(parameters|params)[\s:]+(greater than|less than|equal to|>|<|=)\s*(\d+[KkMmBbTt]?)",
            "date": r"(created|modified|updated)[\s:]+(before|after|between|since)\s+([a-zA-Z0-9_-]+)"
        }

        self.limit_pattern = r"(limit|top|first)\s+(\d+)"
        self.sort_pattern = r"(sort|order)\s+(by|on)\s+([a-zA-Z_]+)\s+(ascending|descending|asc|desc)?"

        # Model name detection - common model families
        self.model_families = [
            "transformer", "gpt", "bert", "t5", "llama", "clip",
            "stable diffusion", "dalle", "cnn", "resnet", "vit",
            "swin", "yolo", "diffusion", "vae", "gan", "bard",
            "mistral", "gemini", "baichuan", "claude", "ernie",
            "chatglm", "falcon", "phi", "qwen", "yi", "bloom", "dqn"
        ]

        # Year extraction patterns with context
        self.created_year_patterns = [
            r"created\s+in\s+(?:the\s+year\s+)?(20\d{2})",
            r"from\s+(?:the\s+year\s+)?(20\d{2})",
            r"developed\s+in\s+(?:the\s+year\s+)?(20\d{2})",
            r"implemented\s+in\s+(?:the\s+year\s+)?(20\d{2})",
            r"built\s+in\s+(?:the\s+year\s+)?(20\d{2})",
            r"made\s+in\s+(?:the\s+year\s+)?(20\d{2})",
            r"dating\s+from\s+(?:the\s+year\s+)?(20\d{2})",
            r"models\s+from\s+(?:the\s+year\s+)?(20\d{2})",
            r"in\s+(?:the\s+year\s+)?(20\d{2})",
            r"year[:\s]+(20\d{2})"
        ]

    def parse_query(self, query_text: str) -> Dict[str, Any]:
        """
        Parse a query to determine intent and parameters.

        Args:
            query_text: The raw query text from the user

        Returns:
            A dictionary containing:
                - intent: The classified intent as a string (not enum)
                - type: Same as intent for backward compatibility
                - parameters: Dictionary of extracted parameters
                - processed_query: The preprocessed query text
        """
        self.logger.debug(f"Parsing query: {query_text}")

        # Preprocess the query
        processed_query = self.preprocess_query(query_text)

        # Classify intent and reason (if any)
        intent, reason = self.classify_intent(query_text)

        # Extract parameters
        parameters = self.extract_parameters(query_text, intent)

        # Convert intent enum to string value for serialization
        intent_str = intent.value if hasattr(intent, 'value') else str(intent)

        result = {
            "intent": intent_str,
            "reason": reason,
            "type": intent_str,  # Add type for backward compatibility
            "parameters": parameters,
            "processed_query": processed_query
        }

        self.logger.info(f"Query parsed: {intent_str} with {len(parameters)} parameters")
        self.logger.debug(f"Parsed result: {result}")

        return result

    def classify_intent(self, query_text: str) -> Union[QueryIntent, tuple[QueryIntent, str]]:
        """
        Classify the intent of a query using the LLM. Rule-based logic is fully embedded in the prompt above.
        """
        try:
            # Initialize LangChain
            system_message = SystemMessage(content=QueryPathPromptManager.get_system_prompt_for_intent_classification())
            human_message = HumanMessagePromptTemplate.from_template("User query: {query}")
            # assemble intent classification prompt
            intent_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
            intent_chain = intent_prompt | self.langchain_llm
            self.logger.info("LangChain for intent parsing initialized successfully")

            # Use invoke() to run the prompt/LLM sequence
            raw_response = intent_chain.invoke({"query": query_text})
            if isinstance(raw_response, dict):
                raw = raw_response.get("text", "")
            else:
                raw = str(raw_response).strip()

            # Extract JSON blob
            match = re.search(r'{.*}', raw, re.DOTALL)
            if not match:
                raise ValueError("No JSON object found in LLM response")
            result = match.group(0)

            import json
            parsed = json.loads(result)
            intent = QueryIntent(parsed["intent"])
            reason = parsed.get("reason", "")
            return intent, reason

        except Exception as e:
            self.logger.error(f"Intent classification failed: {e}")

        # Fallback default to retrieval
        return QueryIntent.RETRIEVAL, "Defaulting to retrieval intent."

    def extract_parameters(self, query_text: str, intent: Optional[QueryIntent] = None) -> Dict[str, Any]:
        """
        Extract parameters from a query based on its intent.

        Args:
            query_text: The query text to extract from
            intent: The query intent, if already classified

        Returns:
            Dictionary of extracted parameters
        """
        if intent is None:
            intent, _ = self.classify_intent(query_text)

        parameters: Dict[str, Any] = {}
        query_lower = query_text.lower()

        # 1) Build filters (month, year, model_id)
        filters = self._build_date_and_model_filters(query_text, query_lower)
        if filters:
            parameters["filters"] = filters

        # 2) Extract generic limit and sort_by
        limit_val = self._extract_limit(query_lower, parameters)
        if limit_val is not None:
            parameters["limit"] = limit_val

        sort_val = self._extract_sort_by(query_lower)
        if sort_val:
            parameters["sort_by"] = sort_val

        # 3) Always extract NER filters via LLM
        parameters["ner_filters"] = self._extract_entities_with_llm(query_text)
        print(f"ner_filters: {parameters['ner_filters']}")

        # 4) Intent-specific image parameters
        if intent == QueryIntent.IMAGE_SEARCH:
            valid_model_ids = self._extract_model_id_mentions(query_text)
            image_params = self._extract_image_parameters(query_text, valid_model_ids)
            parameters.update(image_params)

        return parameters

    def _build_date_and_model_filters(self, query_text: str, query_lower: str) -> Dict[str, Any]:
        """
        Construct a 'filters' dict for created_month, created_year, and model_id if present.
        """
        filters: Dict[str, Any] = {}

        # Month detection
        months = [
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december"
        ]
        for month in months:
            if month in query_lower:
                filters["created_month"] = month.capitalize()
                break

        # Year detection using precompiled patterns
        for pattern in self.created_year_patterns:
            match = re.search(pattern, query_lower)
            if match:
                filters["created_year"] = match.group(1)
                break

        # Model ID mention detection
        valid_model_ids = self._extract_model_id_mentions(query_text)
        if valid_model_ids:
            filters["model_id"] = valid_model_ids[0] if len(valid_model_ids) == 1 else valid_model_ids

        return filters

    def _extract_limit(self, query_lower: str, params: Dict[str, Any]) -> Optional[int]:
        """
        Return an integer limit if the query contains a 'limit/ top/ first N' pattern.
        """
        match = re.search(self.limit_pattern, query_lower)
        return int(match.group(2)) if match else None

    def _extract_sort_by(self, query_lower: str) -> Optional[Dict[str, str]]:
        """
        Return a {'field': ..., 'order': ...} dict if the query contains a sort pattern.
        """
        match = re.search(self.sort_pattern, query_lower)
        if not match:
            return None
        return {
            "field": match.group(3),
            "order": match.group(4) or "descending"
        }

    def _extract_entities_with_llm(self, query_text: str, max_retries: int = 5) -> Dict[str, Any]:
        """
        Use LangChain chat‐style LLM to extract entities from query text with enhanced reasoning,
        strict JSON schema, and separated system vs. human roles.
        """
        # default structure to return on failure
        default_entities = {
            "architecture": {"value": "N/A", "is_positive": True},
            "dataset": {"name": {"value": "N/A", "is_positive": True}},
            "training_config": {
                "batch_size": {"value": "N/A", "is_positive": True},
                "learning_rate": {"value": "N/A", "is_positive": True},
                "optimizer": {"value": "N/A", "is_positive": True},
                "epochs": {"value": "N/A", "is_positive": True},
                "hardware_used": {"value": "N/A", "is_positive": True},
            },
        }

        def _is_valid_schema(entities: Dict[str, Any]) -> bool:
            required = [
                ("architecture", "value"), ("architecture", "is_positive"),
                ("dataset", "value"), ("dataset", "is_positive"),
                ("training_config", "batch_size", "value"), ("training_config", "batch_size", "is_positive"),
                ("training_config", "learning_rate", "value"), ("training_config", "learning_rate", "is_positive"),
                ("training_config", "optimizer", "value"), ("training_config", "optimizer", "is_positive"),
                ("training_config", "epochs", "value"), ("training_config", "epochs", "is_positive"),
                ("training_config", "hardware_used", "value"), ("training_config", "hardware_used", "is_positive"),
            ]
            for path in required:
                d = entities
                for key in path:
                    if not isinstance(d, dict) or key not in d:
                        return False
                    d = d[key]
            return True

        try:
            # Initialize LangChain
            system_message = SystemMessage(content=QueryPathPromptManager.get_system_prompt_for_ner_parsing())

            # human message: only the user’s query
            human_message = HumanMessagePromptTemplate.from_template("User query: {query}")

            # assemble chat prompt
            chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
            entity_chain = chat_prompt | self.langchain_llm
            self.logger.info("LangChain for NER parsing initialized successfully")

            last_exception = None
            for attempt in range(1, max_retries + 1):
                try:
                    # use the chain to invoke with the {"query": ...} input
                    raw_response = entity_chain.invoke({"query": query_text})

                    # extract text from the chain's response
                    if hasattr(raw_response, "content"):
                        raw = raw_response.content
                    elif isinstance(raw_response, dict):
                        raw = raw_response.get("text", "")
                    else:
                        raw = str(raw_response).strip()

                    # strip any LLM “thinking” dumps
                    raw = re.sub(r'<thinking>.*?</thinking>', '', raw, flags=re.DOTALL)
                    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL)

                    # grab only the first balanced JSON-looking chunk
                    match = re.search(r'{.*}', raw, re.DOTALL)
                    if not match:
                        raise ValueError("No JSON object found in LLM response")

                    entities = json.loads(match.group(0))
                    self._convert_numeric_values(entities)

                    print(f"entities: {entities}")

                    if _is_valid_schema(entities):
                        return entities
                    else:
                        raise ValueError(f"Schema validation failed on attempt {attempt}")

                except Exception as e:
                    print(f"[WARNING] Attempt {attempt}/{max_retries} failed: {e}")
                    last_exception = e

            print(f"[ERROR] Entity extraction failed after {max_retries} attempts: {last_exception}")
            return default_entities

        except Exception as e:
            print(f"[ERROR] Entity extraction with LLM failed: {e}")
            return default_entities

    @staticmethod
    def _convert_numeric_values(entities: Dict[str, Any]) -> None:
        """
        Convert string numeric values to actual numeric types in the entities dictionary
        with the new nested structure that includes is_positive flags.
        """
        if "training_config" not in entities:
            return

        config = entities["training_config"]
        QueryParser._convert_int_field(config, "batch_size")
        QueryParser._convert_float_field(config, "learning_rate")
        QueryParser._convert_int_field(config, "epochs")

    @staticmethod
    def _convert_int_field(config: Dict[str, Any], field_name: str) -> None:
        """
        If field_name exists in config and its value is not "N/A",
        extract the first integer substring and replace the value.
        """
        if field_name not in config or config[field_name].get("value") == "N/A":
            return

        value = config[field_name].get("value")
        try:
            numeric_part = re.search(r"\d+", str(value))
            if numeric_part:
                config[field_name]["value"] = int(numeric_part.group(0))
        except (ValueError, TypeError):
            pass

    @staticmethod
    def _convert_float_field(config: Dict[str, Any], field_name: str) -> None:
        """
        If field_name exists in config and its value is not "N/A",
        extract a float (including scientific notation) and replace the value.
        """
        if field_name not in config or config[field_name].get("value") == "N/A":
            return

        value = config[field_name].get("value")
        try:
            text = str(value)
            if "e" in text.lower():
                config[field_name]["value"] = float(text)
            else:
                numeric_part = re.search(r"(\d*\.)?\d+", text)
                if numeric_part:
                    config[field_name]["value"] = float(numeric_part.group(0))
        except (ValueError, TypeError):
            pass

    def _extract_model_id_mentions(self, query_text: str) -> List[str]:
        """
        Extract mentions of model IDs or names from query text.

        Returns:
            List of valid model identifiers (excluding known dataset names and architectures)
        """
        model_ids = []

        # Common datasets to exclude from model_id
        common_datasets = {
            "cifar", "cifar10", "cifar-10", "cifar100", "imagenet", "mnist",
            "fashion-mnist", "coco", "cityscapes", "voc", "svhn", "celeba",
            "librispeech", "wikitext", "squad", "glue", "webtext", "laion", "ms coco",
            "stl", "stl-10", "oxford", "oxford 102"
        }

        # Generic architectures to exclude
        generic_arch = {
            "cnn", "rnn", "lstm", "transformer", "gan", "vae", "mlp",
            "diffusion", "autoencoder", "bert", "gpt", "variational",
            "convolutional", "neural network", "deep learning", "dqn"
            "recurrent", "attention", "generative adversarial"
        }

        model_id_pattern = re.compile(
            self.model_id_pattern,
            flags=re.IGNORECASE
        )
        # Try to extract explicit model_id mentions
        for match in model_id_pattern.finditer(query_text):
            model_id = match.group(1)  # e.g. "Multiplication_scriptRNN_ReversedInputString"
            if model_id.lower() not in generic_arch and model_id.lower() not in common_datasets:
                model_ids.append(model_id)
        print(f"Extracted model id(s) {model_ids} from query {query_text}")

        # Deduplicate
        return list(set(model_ids))

    def _extract_image_parameters(self, query_text: str, valid_model_ids: list) -> Dict[str, Any]:
        """
        Extract image search-specific parameters.

        Args:
            query_text: The query text to extract from
            valid_model_ids: List of valid model IDs (for fallback)

        Returns:
            Dictionary of image search parameters
        """
        params: Dict[str, Any] = {}
        ql = query_text.lower()

        # 1) Determine search_type via small matchers and extractors
        if self._matches_highest_epoch(ql):
            params["search_type"] = "highest_epoch"
        elif self._matches_epoch(ql):
            self._extract_epoch_info(ql, params)
        elif self._matches_tag(ql):
            self._extract_tag_info(ql, params)
        elif self._matches_color(ql):
            self._extract_color_info(ql, params)
        elif self._matches_date(ql):
            self._extract_date_info(ql, params)
        elif self._matches_content(ql):
            self._extract_content_info(ql, params)
        elif valid_model_ids:
            params["search_type"] = "model_id"
        else:
            params["search_type"] = "similarity"

        # 2) Extract optional prompt, style, resolution, and limit
        self._extract_prompt_terms(ql, params)
        self._extract_style_tags(ql, params)
        self._extract_resolution(ql, params)
        self._extract_limit(ql, params)

        return params

    @staticmethod
    def _matches_highest_epoch(ql: str) -> bool:
        return bool(re.search(r"highest\s+epoch|latest\s+epoch", ql))

    @staticmethod
    def _matches_epoch(ql: str) -> bool:
        return bool(re.search(r"epoch\s*[=:]\s*\d+", ql) or re.search(r"from\s+epoch\s+\d+", ql))

    def _extract_epoch_info(self, ql: str, params: Dict[str, Any]) -> None:
        """
        Populate params["search_type"] = "epoch" and extract integer epoch into params["epoch"].
        """
        params["search_type"] = "epoch"
        epoch_match = re.search(r"epoch\s*[=:]\s*(\d+)", ql) or re.search(r"from\s+epoch\s+(\d+)", ql)
        if epoch_match:
            params["epoch"] = int(epoch_match.group(1))

    @staticmethod
    def _matches_tag(ql: str) -> bool:
        return bool(re.search(r"tags?\s*[=:]\s*|with\s+tags?\s+", ql))

    def _extract_tag_info(self, ql: str, params: Dict[str, Any]) -> None:
        """
        Populate params for tag search:
          - params["search_type"] = "tag"
          - params["tags"] = List[str]
          - params["require_all"] = bool
        """
        params["search_type"] = "tag"
        # Match either tag=: "a,b" or with tags "a,b"
        tags_pattern = r'(?:tags?\s*[=:]\s*|with\s+tags?\s+)"?([^"]+)"?'
        match = re.search(tags_pattern, ql)
        if not match:
            return
        tags_str = match.group(1) or match.group(2) or ""
        # Split on commas or "and"
        tags = re.split(r",|\sand\s", tags_str)
        params["tags"] = [t.strip() for t in tags if t.strip()]
        params["require_all"] = "all" in ql and "tags" in ql

    @staticmethod
    def _matches_color(ql: str) -> bool:
        return bool(re.search(r"colors?\s*[=:]\s*|with\s+colors?\s+", ql))

    def _extract_color_info(self, ql: str, params: Dict[str, Any]) -> None:
        """
        Populate params for color search:
          - params["search_type"] = "color"
          - params["colors"] = List[str]
        """
        params["search_type"] = "color"
        colors_pattern = r'(?:colors?\s*[=:]\s*|with\s+colors?\s+)"?([^"]+)"?'
        match = re.search(colors_pattern, ql)
        if not match:
            return
        colors_str = match.group(1) or match.group(2) or ""
        colors = re.split(r",|\sand\s", colors_str)
        params["colors"] = [c.strip() for c in colors if c.strip()]

    @staticmethod
    def _matches_date(ql: str) -> bool:
        return bool(re.search(r"date\s*[=:]\s*|created\s+(on|in)\s+", ql))

    def _extract_date_info(self, ql: str, params: Dict[str, Any]) -> None:
        """
        Populate params for date search:
          - params["search_type"] = "date"
          - params["date_filter"] = {"created_year": "YYYY", "created_month": "MM"}
        """
        params["search_type"] = "date"
        date_filter: Dict[str, str] = {}

        # Year (e.g., 2021, 2022, ...)
        year_match = re.search(r"(20\d{2})", ql)
        if year_match:
            date_filter["created_year"] = year_match.group(1)

        # Month names → numeric string
        months = [
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december"
        ]
        for idx, month in enumerate(months, 1):
            if month in ql:
                date_filter["created_month"] = str(idx).zfill(2)
                break

        params["date_filter"] = date_filter

    @staticmethod
    def _matches_content(ql: str) -> bool:
        return bool(re.search(r"content\s*[=:]\s*|subject\s*[=:]\s*|scene\s*[=:]\s*", ql))

    def _extract_content_info(self, ql: str, params: Dict[str, Any]) -> None:
        """
        Populate params for content search:
          - params["search_type"] = "content"
          - params["content_filter"] = {"subject_type": str, "scene_type": str}
        """
        params["search_type"] = "content"
        content_filter: Dict[str, str] = {}

        subject_match = re.search(r"subject\s*[=:]\s*\"?([^\"]+)\"?", ql)
        if subject_match:
            content_filter["subject_type"] = subject_match.group(1).strip()

        scene_match = re.search(r"scene\s*[=:]\s*\"?([^\"]+)\"?", ql)
        if scene_match:
            content_filter["scene_type"] = scene_match.group(1).strip()

        params["content_filter"] = content_filter

    def _extract_prompt_terms(self, ql: str, params: Dict[str, Any]) -> None:
        """
        If the query includes a prompt/text indicator, extract and store it:
          - params["prompt_terms"]
          - If search_type == "similarity", also set params["query_text"]
        """
        prompt_pattern = r"(prompt|prompts|text)[:\s]+[\"']?([\w\s,]+)[\"']?"
        match = re.search(prompt_pattern, ql)
        if not match:
            return
        prompt = match.group(2).strip()
        params["prompt_terms"] = prompt
        if params.get("search_type") == "similarity":
            params["query_text"] = prompt

    def _extract_style_tags(self, ql: str, params: Dict[str, Any]) -> None:
        """
        Extract style tags from the query if present:
          - params["style_tags"] = List[str]
        """
        style_pattern = r"(style|type|category|look)[:\s]+[\"']?([\w\s,]+)[\"']?"
        match = re.search(style_pattern, ql)
        if not match:
            return
        raw_styles = match.group(2)
        styles = re.split(r",|\sand\s", raw_styles)
        params["style_tags"] = [s.strip() for s in styles if s.strip()]

    def _extract_resolution(self, ql: str, params: Dict[str, Any]) -> None:
        """
        Extract resolution (width x height) if specified:
          - params["resolution"] = {"width": int, "height": int}
        """
        resolution_pattern = r"(resolution|size|dimensions)[:\s]+(\d+)\s*[x×]\s*(\d+)"
        match = re.search(resolution_pattern, ql)
        if not match:
            return
        params["resolution"] = {
            "width": int(match.group(2)),
            "height": int(match.group(3))
        }

    def _extract_limit(self, ql: str, params: Dict[str, Any]) -> None:
        """
        Extract a limit for number of results if specified, else default to 10:
          - params["limit"]
        """
        limit_pattern = r"(limit|top|first)\s+(\d+)"
        match = re.search(limit_pattern, ql)
        if match:
            params["limit"] = int(match.group(2))
        else:
            params["limit"] = 10

    def preprocess_query(self, query_text: str) -> str:
        """
        Preprocess a natural language query to optimize for AI model search and analysis.
        """
        clean_text = query_text.strip()
        doc = self.nlp(clean_text)

        processed_tokens: List[str] = []
        skip_indices: Set[int] = set()

        # STAGE 1: Preserve named entities
        self._preserve_named_entities(doc, processed_tokens, skip_indices)

        # STAGE 2: Preserve model-related noun phrases
        self._preserve_model_noun_phrases(doc, processed_tokens, skip_indices)

        # STAGE 3: Process remaining tokens
        self._process_remaining_tokens(doc, processed_tokens, skip_indices)

        return " ".join(processed_tokens)

    def _preserve_named_entities(
            self,
            doc: spacy.tokens.Doc,
            processed_tokens: List[str],
            skip_indices: Set[int]
    ) -> None:
        """
        Find entities with labels PRODUCT, ORG, GPE, PERSON, WORK_OF_ART,
        mark their token indices as skipped, and add the full text to processed_tokens.
        """
        for ent in doc.ents:
            if ent.label_ in {"PRODUCT", "ORG", "GPE", "PERSON", "WORK_OF_ART"}:
                for i in range(ent.start, ent.end):
                    skip_indices.add(i)
                processed_tokens.append(ent.text)

    def _preserve_model_noun_phrases(
            self,
            doc: spacy.tokens.Doc,
            processed_tokens: List[str],
            skip_indices: Set[int]
    ) -> None:
        """
        For each noun chunk containing any term in self.model_families,
        mark its token indices as skipped (if not already) and add the chunk text.
        """
        for chunk in doc.noun_chunks:
            text_lower = chunk.text.lower()
            contains_model_family = any(
                family in text_lower for family in self.model_families
            )
            if not contains_model_family:
                continue

            # If at least one token in this chunk isn't yet skipped, preserve it
            if any(i not in skip_indices for i in range(chunk.start, chunk.end)):
                for i in range(chunk.start, chunk.end):
                    skip_indices.add(i)
                processed_tokens.append(chunk.text)

    def _process_remaining_tokens(
            self,
            doc: spacy.tokens.Doc,
            processed_tokens: List[str],
            skip_indices: Set[int]
    ) -> None:
        """
        For each token not in skip_indices, filter out stopwords, punctuation, spaces,
        and single-character tokens; then add its lowercase lemma.
        """
        for i, token in enumerate(doc):
            if i in skip_indices:
                continue

            if (
                    token.is_stop
                    or token.is_punct
                    or token.is_space
                    or len(token.text.strip()) <= 1
            ):
                continue

            processed_tokens.append(token.lemma_.lower())

