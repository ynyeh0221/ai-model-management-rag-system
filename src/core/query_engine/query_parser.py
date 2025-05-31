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
from typing import Dict, List, Any, Optional, Union

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
        >>> print(result["intent"])  # 'retrieval'
        >>> print(result["parameters"]["limit"])  # 5
    """

    def __init__(self, nlp_model: str = "en_core_web_sm", llm_model_name: str = "deepseek-r1:7b"):
        """
        Initialize the QueryParser with necessary NLP components.

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

        parameters = {}
        query_lower = query_text.lower()

        # Initialize or retrieve existing filters
        filters = parameters.get("filters", {})

        # Defensive enhancement: always try to detect month/year references
        months = [
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december"
        ]
        for month in months:
            if month in query_lower:
                filters["created_month"] = month.capitalize()
                break

        # Extract year using context-aware patterns
        for pattern in self.created_year_patterns:
            year_match = re.search(pattern, query_lower)
            if year_match:
                filters["created_year"] = year_match.group(1)
                break

        valid_model_ids = self._extract_model_id_mentions(query_text)
        if valid_model_ids:
            filters["model_id"] = valid_model_ids[0] if len(valid_model_ids) == 1 else valid_model_ids

        if filters:
            parameters["filters"] = filters

        # Result limit
        limit_match = re.search(self.limit_pattern, query_lower)
        if limit_match:
            parameters["limit"] = int(limit_match.group(2))

        # Sort
        sort_match = re.search(self.sort_pattern, query_lower)
        if sort_match:
            parameters["sort_by"] = {
                "field": sort_match.group(3),
                "order": sort_match.group(4) if sort_match.group(4) else "descending"
            }

        # Extract architecture, dataset, and training configuration using LLM
        # Store these in a separate ner_filters parameter
        parameters["ner_filters"] = self._extract_entities_with_llm(query_text)
        print(f"ner_filters: {parameters['ner_filters']}")

        # Intent-specific
        if intent == QueryIntent.IMAGE_SEARCH:
            parameters.update(self._extract_image_parameters(query_text, valid_model_ids))

        return parameters

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

        Args:
            entities: Dictionary of entities extracted by the LLM
        """
        # Convert training config numeric values
        if "training_config" in entities:
            config = entities["training_config"]

            # Convert batch size to integer
            if "batch_size" in config and config["batch_size"]["value"] != "N/A":
                try:
                    # Handle potential formatted strings like "32" or "32 samples"
                    batch_size_str = str(config["batch_size"]["value"])
                    # Extract the numeric part
                    numeric_part = re.search(r'\d+', batch_size_str)
                    if numeric_part:
                        config["batch_size"]["value"] = int(numeric_part.group(0))
                except (ValueError, TypeError):
                    pass

            # Convert learning rate to float
            if "learning_rate" in config and config["learning_rate"]["value"] != "N/A":
                try:
                    # Handle potential formatted strings like "0.001" or "1e-3"
                    lr_str = str(config["learning_rate"]["value"])
                    # Check for scientific notation
                    if 'e' in lr_str.lower():
                        config["learning_rate"]["value"] = float(lr_str)
                    else:
                        # Extract the numeric part
                        numeric_part = re.search(r'([0-9]*[.])?[0-9]+', lr_str)
                        if numeric_part:
                            config["learning_rate"]["value"] = float(numeric_part.group(0))
                except (ValueError, TypeError):
                    pass

            # Convert epochs to integer
            if "epochs" in config and config["epochs"]["value"] != "N/A":
                try:
                    # Handle potential formatted strings like "100" or "100 epochs"
                    epochs_str = str(config["epochs"]["value"])
                    # Extract the numeric part
                    numeric_part = re.search(r'\d+', epochs_str)
                    if numeric_part:
                        config["epochs"]["value"] = int(numeric_part.group(0))
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
        Extract image search specific parameters.

        Args:
            query_text: The query text to extract from

        Returns:
            Dictionary of image search parameters
        """
        params = {}
        query_lower = query_text.lower()

        # Determine search type based on query patterns
        if re.search(r"highest\s+epoch|latest\s+epoch", query_lower):
            params["search_type"] = "highest_epoch"
        elif re.search(r"epoch\s*[=:]\s*\d+", query_lower) or re.search(r"from\s+epoch\s+\d+", query_lower):
            params["search_type"] = "epoch"
            # Extract epoch number
            epoch_match = re.search(r"epoch\s*[=:]\s*(\d+)", query_lower) or re.search(r"from\s+epoch\s+(\d+)",
                                                                                       query_lower)
            if epoch_match:
                params["epoch"] = int(epoch_match.group(1))
        elif re.search(r"tag[s]?\s*[=:]\s*|with\s+tags?\s+", query_lower):
            params["search_type"] = "tag"
            # Extract tags
            tags_pattern = r"tag[s]?\s*[=:]\s*\"?([^\"]+)\"?|with\s+tags?\s+\"?([^\"]+)\"?"
            tags_match = re.search(tags_pattern, query_lower)
            if tags_match:
                # Get whichever group matched
                tags_str = tags_match.group(1) if tags_match.group(1) else tags_match.group(2)
                # Split by commas or 'and'
                tags = re.split(r',|\sand\s', tags_str)
                params["tags"] = [tag.strip() for tag in tags if tag.strip()]
                params["require_all"] = "all" in query_lower and "tags" in query_lower
        elif re.search(r"color[s]?\s*[=:]\s*|with\s+color[s]?\s+", query_lower):
            params["search_type"] = "color"
            # Extract colors
            colors_pattern = r"color[s]?\s*[=:]\s*\"?([^\"]+)\"?|with\s+color[s]?\s+\"?([^\"]+)\"?"
            colors_match = re.search(colors_pattern, query_lower)
            if colors_match:
                # Get whichever group matched
                colors_str = colors_match.group(1) if colors_match.group(1) else colors_match.group(2)
                # Split by commas or 'and'
                colors = re.split(r',|\sand\s', colors_str)
                params["colors"] = [color.strip() for color in colors if color.strip()]
        elif re.search(r"date\s*[=:]\s*|created\s+(on|in)\s+", query_lower):
            params["search_type"] = "date"
            # Extract date components
            date_filter = {}

            # Look for year
            year_match = re.search(r"(20\d{2})", query_lower)
            if year_match:
                date_filter["created_year"] = year_match.group(1)

            # Look for month
            months = ["january", "february", "march", "april", "may", "june",
                      "july", "august", "september", "october", "november", "december"]
            for i, month in enumerate(months, 1):
                if month in query_lower:
                    date_filter["created_month"] = str(i).zfill(2)  # "01" for January, etc.
                    break

            params["date_filter"] = date_filter
        elif re.search(r"content\s*[=:]\s*|subject\s*[=:]\s*|scene\s*[=:]\s*", query_lower):
            params["search_type"] = "content"
            # Extract content filter components
            content_filter = {}

            # Subject type
            subject_match = re.search(r"subject\s*[=:]\s*\"?([^\"]+)\"?", query_lower)
            if subject_match:
                content_filter["subject_type"] = subject_match.group(1).strip()

            # Scene type
            scene_match = re.search(r"scene\s*[=:]\s*\"?([^\"]+)\"?", query_lower)
            if scene_match:
                content_filter["scene_type"] = scene_match.group(1).strip()

            params["content_filter"] = content_filter
        elif valid_model_ids is not None and len(valid_model_ids) > 0:
            params["search_type"] = "model_id"
        else:
            # Default to similarity search
            params["search_type"] = "similarity"

        # Extract prompt terms (for similarity search)
        prompt_pattern = r"(prompt|prompts|text)[:\s]+[\"']?([\w\s,]+)[\"']?"
        match = re.search(prompt_pattern, query_lower)
        if match:
            params["prompt_terms"] = match.group(2).strip()

            # If search_type is similarity, set query_text as well
            if params.get("search_type") == "similarity":
                params["query_text"] = match.group(2).strip()

        # Extract style tags
        style_pattern = r"(style|type|category|look)[:\s]+[\"']?([\w\s,]+)[\"']?"
        match = re.search(style_pattern, query_lower)
        if match:
            # Split by commas and clean up
            styles = re.split(r',|\sand\s', match.group(2))
            params["style_tags"] = [style.strip() for style in styles if style.strip()]

        # Extract resolution preference
        resolution_pattern = r"(resolution|size|dimensions)[:\s]+(\d+)\s*[x×]\s*(\d+)"
        match = re.search(resolution_pattern, query_lower)
        if match:
            params["resolution"] = {
                "width": int(match.group(2)),
                "height": int(match.group(3))
            }

        # Extract limit for results
        limit_pattern = r"(limit|top|first)\s+(\d+)"
        match = re.search(limit_pattern, query_lower)
        if match:
            params["limit"] = int(match.group(2))
        else:
            # Default limit
            params["limit"] = 10

        return params

    def preprocess_query(self, query_text: str) -> str:
        """
        Preprocess a natural language query to optimize for AI model search and analysis.

        This function performs intelligent multi-stage text processing:
        1. Preserves named entities (PRODUCT, ORG, GPE, PERSON, WORK_OF_ART)
        2. Preserves noun phrases containing model family names (e.g., "transformer model")
        3. Standardizes remaining words through lemmatization and filters out noise

        The processing pipeline prioritizes technical accuracy while improving search consistency.

        Args:
            query_text: The raw query text from the user

        Returns:
            str: Preprocessed query text optimized for model search and parameter extraction

        Examples:
            >>> parser.preprocess_query("Show me GPT-4 models trained by OpenAI using transformers")
            'GPT-4 OpenAI transformer model train'

            >>> parser.preprocess_query("Find the best performing BERT models from 2023")
            'BERT model best perform from 2023'
        """
        # Basic text cleaning
        clean_text = query_text.strip()

        # Parse with spaCy
        doc = self.nlp(clean_text)

        # Initialize processing variables
        processed_tokens = []
        skip_indices = set()

        # STAGE 1: Preserve important named entities
        for entity in doc.ents:
            if entity.label_ in ["PRODUCT", "ORG", "GPE", "PERSON", "WORK_OF_ART"]:
                # Mark all tokens in this entity as processed
                for i in range(entity.start, entity.end):
                    skip_indices.add(i)
                # Keep the entity text as-is
                processed_tokens.append(entity.text)

        # STAGE 2: Preserve model-related noun phrases
        for noun_phrase in doc.noun_chunks:
            # Check if this phrase contains any model family name
            has_model_term = any(
                family in noun_phrase.text.lower()
                for family in self.model_families
            )

            if has_model_term:
                # Check if we've already processed all tokens in this phrase
                tokens_already_processed = all(
                    i in skip_indices
                    for i in range(noun_phrase.start, noun_phrase.end)
                )

                if not tokens_already_processed:
                    # Mark all tokens in this phrase as processed
                    for i in range(noun_phrase.start, noun_phrase.end):
                        skip_indices.add(i)
                    # Keep the phrase text as-is
                    processed_tokens.append(noun_phrase.text)

        # STAGE 3: Process remaining tokens
        for i, token in enumerate(doc):
            if i not in skip_indices:
                # Filter out noise tokens
                is_meaningful = (
                        not token.is_stop and  # Not a stopword
                        not token.is_punct and  # Not punctuation
                        not token.is_space and  # Not whitespace
                        len(token.text.strip()) > 1  # Not a single character
                )

                if is_meaningful:
                    # Add the legitimatized, lowercase form
                    processed_tokens.append(token.lemma_.lower())

        # Combine all processed tokens into a single string
        return " ".join(processed_tokens)
