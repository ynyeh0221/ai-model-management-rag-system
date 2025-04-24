import logging
import re
from enum import Enum
from typing import Dict, List, Any, Optional, Union

import nltk
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Try to import sentence-transformers if installed
try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Try to import Hugging Face transformers and onnxruntime for local LLM inference
try:
    import transformers
    import onnxruntime

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# Define intent types as an Enum for type safety
class QueryIntent(Enum):
    RETRIEVAL = "retrieval"  # Basic information retrieval
    COMPARISON = "comparison"  # Model comparison
    NOTEBOOK = "notebook"  # Notebook generation
    IMAGE_SEARCH = "image_search"  # Image search/retrieval
    METADATA = "metadata"  # Metadata-specific queries
    UNKNOWN = "unknown"  # Unknown/ambiguous intent


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
        use_langchain: Boolean indicating whether to use LangChain for intent classification.
        intent_chain: LangChain chain for intent classification (when use_langchain is True).
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

    def __init__(self, nlp_model: str = "en_core_web_sm", use_langchain: bool = True, llm_model_name: str = "deepseek-llm:7b"):
        """
        Initialize the QueryParser with necessary NLP components.

        Args:
            nlp_model: The spaCy model to use for NLP tasks
            use_langchain: Whether to use LangChain for enhanced parsing
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

        # Initialize LangChain components if enabled
        self.use_langchain = use_langchain
        if use_langchain:
            try:
                # Adjust this if your version requires a submodule import.
                from langchain_ollama import OllamaLLM
                from langchain.chains import LLMChain
                from langchain.prompts import PromptTemplate

                # Define a prompt template for intent classification
                intent_template = """
                You are an expert system that classifies user queries about AI models into one of these categories:
                - retrieval: asking for general information (e.g. "What is GPT-4?", "Show me models trained on ImageNet")
                - comparison: queries that compare two or more models (must mention at least two model names and comparison keywords like "compare", "vs", "better than")
                - notebook: requests to generate or work with analysis code/notebooks (keywords: create, generate, notebook, Colab, script)
                - image_search: looking for images generated by models, including epoch-based or tag-based filtering (keywords: image, picture, from epoch, tags, colors)
                - metadata: inquiries about model metadata such as creation date, last modified, schema or properties (mentions of months, years, dates, metadata, fields, properties)

                IMPORTANT RULES:
                1. Only classify as comparison if at least two model names are mentioned (exclude dataset or architecture names) and comparison terms are present.
                2. Month/year references like "created in March" or "models from 2022" must be classified as metadata.
                3. Do NOT classify as comparison solely because dataset names (e.g. CIFAR, MNIST) appear.
                4. If keywords for notebook or image_search appear, choose those intents over retrieval.
                5. Retrieval is the default for any general informational query (keywords: find, get, retrieve, what is, show, details) and for any query not matching other intents.
                6. If the query contains the word "using" followed by any recognized model family or architecture (Transformer, GPT, BERT, T5, LLaMA, CLIP, ResNet, ViT, etc.), classify it as retrieval.
                7. Polite prefixes like "please", "kindly", or "could you" do not change the intent.
                8. Terms like "introduce", "describe", "detail", or "give details" should map to retrieval.
                9. If the query contains the keyword "compare" and at least two explicit model identifiers (e.g., phrases starting with "model id" or known model names), classify as comparison.
                10. If the query contains phrases like 'find image', 'find pictures', 'search for images', 'look for images', or any similar phrase about finding/retrieving images, ALWAYS classify as image_search regardless of other criteria.

                EXAMPLES:
                - Query: "Find models using Transformer" -> retrieval
                - Query: "Please find models using Transformer and introduce their details" -> retrieval
                - Query: "Compare GPT-3 vs. GPT-4" -> comparison
                - Query: "Compare model id Multiplication_scriptRNN_ReversedInputString and model id IMDB_scriptRNN" -> comparison
                - Query: "Show me the properties of models from March" -> metadata
                - Query: "Generate a Colab notebook" -> notebook
                - Query: "Show images from epoch 10 with tags 'sunset'" -> image_search
                - Query: "Find images of models trained on CIFAR-10" -> image_search
                - Query: "Please find images of model id ABC" -> image_search
                - Query: "Can you find pictures generated by any models?" -> image_search

                Query: {query}

                Respond ONLY in JSON format:
                {{  
                  "intent": "<intent_name>",
                  "reason": "<short explanation of why this intent was chosen>"
                }}"""

                self.intent_prompt = PromptTemplate(
                    input_variables=["query"],
                    template=intent_template
                )

                # Use the correct model identifier as expected by Ollama.
                self.langchain_llm = OllamaLLM(model=llm_model_name, temperature=0)

                self.intent_chain = self.intent_prompt | self.langchain_llm

                self.logger.info("LangChain components initialized successfully")
            except Exception as e:
                self.logger.warning(f"LangChain not available: {e}. Falling back to rule-based parsing")
                self.use_langchain = False

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
        if self.use_langchain:
            try:
                # Use invoke() to run the prompt/LLM sequence
                raw_response = self.intent_chain.invoke({"query": query_text})
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
            intent, reason = self.classify_intent(query_text)

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

        # Intent-specific
        if intent == QueryIntent.IMAGE_SEARCH:
            parameters.update(self._extract_image_parameters(query_text, valid_model_ids))

        return parameters

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
                    # Add the lemmatized, lowercase form
                    processed_tokens.append(token.lemma_.lower())

        # Combine all processed tokens into a single string
        return " ".join(processed_tokens)
