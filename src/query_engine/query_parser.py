import logging
import re
from enum import Enum
from typing import Dict, List, Any, Optional

import nltk
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.spatial.distance import cosine

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
    Responsible for intent classification and parameter extraction.
    """

    def __init__(self, nlp_model: str = "en_core_web_sm", use_langchain: bool = True,
                 use_sentence_transformer: bool = True, use_local_llm: bool = False,
                 local_llm_path: str = None):
        """
        Initialize the QueryParser with necessary NLP components.

        Args:
            nlp_model: The spaCy model to use for NLP tasks
            use_langchain: Whether to use LangChain for enhanced parsing
            use_sentence_transformer: Whether to use sentence-transformer for semantic matching
            use_local_llm: Whether to use a local LLM for intent classification and parameter extraction
            local_llm_path: Path to local LLM model
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

        # Initialize semantic matching components
        self.use_sentence_transformer = use_sentence_transformer and SENTENCE_TRANSFORMERS_AVAILABLE
        if self.use_sentence_transformer:
            try:
                # Use a small model suitable for local execution
                self.sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
                self.logger.info("Initialized sentence-transformer model")
            except Exception as e:
                self.logger.warning(f"Could not initialize sentence-transformer: {e}")
                self.use_sentence_transformer = False

        # Initialize local LLM components
        self.use_local_llm = use_local_llm and TRANSFORMERS_AVAILABLE
        if self.use_local_llm:
            try:
                if local_llm_path:
                    # Use specified local model path
                    self.tokenizer = transformers.AutoTokenizer.from_pretrained(local_llm_path)
                    self.local_llm = transformers.pipeline(
                        "text-generation",
                        model=local_llm_path,
                        tokenizer=self.tokenizer,
                        device_map="auto"  # Use GPU if available
                    )
                else:
                    # Use default small model suitable for local execution
                    self.tokenizer = transformers.AutoTokenizer.from_pretrained("distilgpt2")
                    self.local_llm = transformers.pipeline(
                        "text-generation",
                        model="distilgpt2",
                        tokenizer=self.tokenizer,
                        device_map="auto"
                    )
                self.logger.info("Initialized local LLM model")
            except Exception as e:
                self.logger.warning(f"Could not initialize local LLM: {e}")
                self.use_local_llm = False

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
                Classify the following query about AI models into one of these categories:
                - retrieval: Basic information retrieval about models
                - comparison: Comparing multiple models
                - notebook: Generating a notebook for model analysis
                - image_search: Searching for images generated by models
                - metadata: Queries about model metadata
                - unknown: Cannot determine the intent

                Query: {query}

                Intent:
                """

                self.intent_prompt = PromptTemplate(
                    input_variables=["query"],
                    template=intent_template
                )

                # Use the correct model identifier as expected by Ollama.
                self.langchain_llm = OllamaLLM(model="llama3:latest", temperature=0)

                self.intent_chain = self.intent_prompt | self.langchain_llm

                # Define a prompt template for parameter extraction
                param_template = """
                Extract parameters from this query about AI models.
                Return a JSON object with these possible keys:
                - model_ids: List of model IDs mentioned
                - metrics: Performance metrics of interest
                - filters: Any filtering criteria
                - limit: Number of results to return
                - sort_by: Sorting criteria
                - timeframe: Any time constraints

                Only include keys that are relevant to the query.

                Query: {query}

                Parameters:
                """

                self.param_prompt = PromptTemplate(
                    input_variables=["query"],
                    template=param_template
                )

                self.param_chain = self.param_prompt | self.langchain_llm

                self.logger.info("LangChain components initialized successfully")
            except Exception as e:
                self.logger.warning(f"LangChain not available: {e}. Falling back to rule-based parsing")
                self.use_langchain = False

        # Initialize pattern dictionaries for rule-based parsing
        self._init_patterns()

        # Create standard examples for semantic matching
        self._init_semantic_examples()

    def _init_patterns(self):
        """Initialize regex patterns and keywords for rule-based parsing."""
        # Intent classification patterns
        self.intent_patterns = {
            QueryIntent.RETRIEVAL: [
                r"find|get|retrieve|show|display|tell me about|information on|details of",
                r"what (is|are)|how (is|are)|where (is|are)|when (was|were)"
            ],
            QueryIntent.COMPARISON: [
                r"compare|versus|vs\.?|difference between|similarities between|better than",
                r"which (is|are) (better|worse|faster|more accurate)",
                r"(compare|comparing) the (performance|accuracy|results) of"
            ],
            QueryIntent.NOTEBOOK: [
                r"(create|generate|make|build) (a |an )?(notebook|colab|code|script)",
                r"jupyter|analysis script|analysis code",
                r"notebook (for|to) (analyze|explore|compare)"
            ],
            QueryIntent.IMAGE_SEARCH: [
                r"(find|get|retrieve|show|display) (image|picture|photo)",
                r"(generated|created) (by|with|using)",
                r"(show|find|get) (me )?(examples|samples) (of|from)"
            ],
            QueryIntent.METADATA: [
                r"\b(metadata|schema|fields|properties|attributes)\b",
                r"\bwhat (fields|properties|attributes)\b.*",
                r"\b(structure|organization) of\b.*"
            ]
        }

        # Parameter extraction patterns
        self.model_id_pattern = r"(model[_-]?id|model)[:\s]+([a-zA-Z0-9_-]+)"
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
            "chatglm", "falcon", "phi", "qwen", "yi", "bloom"
        ]

    def _init_semantic_examples(self):
        """Initialize standard example queries for semantic matching"""
        self.intent_examples = {
            QueryIntent.RETRIEVAL: [
                "tell me about GPT-4",
                "what is BERT?",
                "show details of Llama model",
                "give me information on transformer architecture",
                "tell me about the features of T5"
            ],
            QueryIntent.COMPARISON: [
                "compare GPT-4 and BERT",
                "what's the difference between Llama and GPT?",
                "how does T5 compare to BERT in accuracy?",
                "which is better, GPT-3 or GPT-4?",
                "compare the performance of transformer models"
            ],
            QueryIntent.NOTEBOOK: [
                "create a notebook for analyzing GPT-4",
                "generate code to evaluate BERT",
                "make a jupyter notebook for model comparison",
                "build an analysis script for diffusion models",
                "generate a colab to explore transformer architecture"
            ],
            QueryIntent.IMAGE_SEARCH: [
                "show images generated by Stable Diffusion",
                "find pictures created with DALL-E",
                "display examples of GAN-generated images",
                "retrieve photos made by diffusion models",
                "get sample images from Midjourney"
            ],
            QueryIntent.METADATA: [
                "what metadata fields does GPT-4 have?",
                "show me the schema of BERT",
                "what properties does Llama model include?",
                "tell me about the structure of transformer metadata",
                "what attributes are stored for diffusion models?"
            ]
        }

        # Pre-compute example embeddings (if sentence-transformer is enabled)
        if self.use_sentence_transformer:
            self.intent_embeddings = {}
            for intent, examples in self.intent_examples.items():
                self.intent_embeddings[intent] = self.sentence_model.encode(examples)

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

        # Classify intent
        intent = self.classify_intent(query_text)

        # Extract parameters
        parameters = self.extract_parameters(query_text, intent)

        # Convert intent enum to string value for serialization
        intent_str = intent.value if hasattr(intent, 'value') else str(intent)

        result = {
            "intent": intent_str,
            "type": intent_str,  # Add type for backward compatibility
            "parameters": parameters,
            "processed_query": processed_query
        }

        self.logger.info(f"Query parsed: {intent_str} with {len(parameters)} parameters")
        self.logger.debug(f"Parsed result: {result}")

        return result

    def classify_intent(self, query_text: str) -> QueryIntent:
        """
        Classify the intent of a query.
        """
        query_lower = query_text.lower()
        model_mentions = self._extract_model_mentions(query_lower)

        # Step 1: Try using semantic similarity (if available)
        if self.use_sentence_transformer:
            try:
                query_embedding = self.sentence_model.encode([query_text])[0]
                best_score = -1
                best_intent = None

                for intent, embeddings in self.intent_embeddings.items():
                    similarities = [1 - cosine(query_embedding, emb) for emb in embeddings]
                    max_similarity = max(similarities)
                    if max_similarity > best_score:
                        best_score = max_similarity
                        best_intent = intent

                if best_score > 0.5:
                    return best_intent
            except Exception as e:
                self.logger.warning(f"Semantic matching failed: {e}")

        # Step 2: Rule-based pattern matching (ordered by priority)
        if len(model_mentions) > 1 and any(
                re.search(p, query_lower) for p in self.intent_patterns[QueryIntent.COMPARISON]):
            return QueryIntent.COMPARISON

        if any(re.search(p, query_lower) for p in self.intent_patterns[QueryIntent.METADATA]):
            return QueryIntent.METADATA

        if any(re.search(p, query_lower) for p in self.intent_patterns[QueryIntent.NOTEBOOK]):
            return QueryIntent.NOTEBOOK

        if any(re.search(p, query_lower) for p in self.intent_patterns[QueryIntent.IMAGE_SEARCH]):
            return QueryIntent.IMAGE_SEARCH

        if any(re.search(p, query_lower) for p in self.intent_patterns[QueryIntent.RETRIEVAL]):
            return QueryIntent.RETRIEVAL

        if model_mentions:
            return QueryIntent.RETRIEVAL

        return QueryIntent.UNKNOWN

    def _nlp_based_intent_classification(self, query_text: str) -> QueryIntent:
        """
        Use NLP techniques to classify intent when rule-based approach is inconclusive.

        Args:
            query_text: The query text to classify

        Returns:
            QueryIntent: The classified intent
        """
        # Parse with spaCy
        doc = self.nlp(query_text)

        # Extract verbs and nouns for intent analysis
        verbs = [token.lemma_ for token in doc if token.pos_ == "VERB"]
        nouns = [token.lemma_ for token in doc if token.pos_ == "NOUN"]

        # Create vocabulary sets for different intents
        comparison_verbs = ["compare", "contrast", "differ", "distinguish", "evaluate", "versus", "vs", "against"]
        comparison_nouns = ["comparison", "difference", "similarity", "distinction", "differential"]

        notebook_verbs = ["create", "generate", "make", "build", "develop", "code", "program", "script"]
        notebook_nouns = ["notebook", "code", "script", "program", "analysis", "jupyter", "colab", "coding"]

        image_verbs = ["show", "display", "find", "get", "generate", "create", "render", "visualize"]
        image_nouns = ["image", "picture", "photo", "visualization", "render", "sample", "example"]

        metadata_nouns = ["metadata", "schema", "field", "property", "attribute", "structure", "configuration",
                          "parameter", "setup"]

        model_nouns = ["model", "transformer", "neural", "network", "ai", "architecture", "system"]

        # Optimized classification logic - weight different types of words
        # Use a more sophisticated scoring mechanism
        intent_scores = {intent: 0 for intent in QueryIntent}

        # Score for comparison intent
        if any(verb in comparison_verbs for verb in verbs):
            intent_scores[QueryIntent.COMPARISON] += 2
        if any(noun in comparison_nouns for noun in nouns):
            intent_scores[QueryIntent.COMPARISON] += 1
        if len(set(nouns)) > 1 and any(noun in model_nouns for noun in nouns):
            intent_scores[QueryIntent.COMPARISON] += 1

        # Score for notebook intent
        if any(verb in notebook_verbs for verb in verbs):
            intent_scores[QueryIntent.NOTEBOOK] += 2
        if any(noun in notebook_nouns for noun in nouns):
            intent_scores[QueryIntent.NOTEBOOK] += 2

        # Score for image search intent
        if any(verb in image_verbs for verb in verbs) and any(noun in image_nouns for noun in nouns):
            intent_scores[QueryIntent.IMAGE_SEARCH] += 3
        elif any(verb in image_verbs for verb in verbs):
            intent_scores[QueryIntent.IMAGE_SEARCH] += 1
        elif any(noun in image_nouns for noun in nouns):
            intent_scores[QueryIntent.IMAGE_SEARCH] += 1

        # Score for metadata intent
        if any(noun in metadata_nouns for noun in nouns):
            intent_scores[QueryIntent.METADATA] += 2

        # Score for retrieval intent (default)
        if any(noun in model_nouns for noun in nouns):
            intent_scores[QueryIntent.RETRIEVAL] += 1

        # Get the intent with the highest score
        best_intent = max(intent_scores.items(), key=lambda x: x[1])

        # If we have a clear winner with a score > 0, return it
        if best_intent[1] > 0:
            return best_intent[0]

        # If still uncertain, default to UNKNOWN
        return QueryIntent.UNKNOWN

    def extract_parameters(self, query_text: str, intent: Optional[QueryIntent] = None) -> Dict[str, Any]:
        """
        Extract parameters from a query based on its intent.

        Args:
            query_text: The query text to extract parameters from
            intent: The query intent, if already classified

        Returns:
            Dictionary of extracted parameters
        """
        if intent is None:
            intent = self.classify_intent(query_text)

        # Try using local LLM for parameter extraction
        if self.use_local_llm:
            try:
                # Construct prompt for parameter extraction
                prompt = f"""
                Extract parameters from this query about AI models.
                Return a JSON object with these possible keys:
                - model_ids: List of model IDs mentioned
                - metrics: Performance metrics of interest
                - filters: Any filtering criteria
                - limit: Number of results to return
                - sort_by: Sorting criteria
                - timeframe: Any time constraints

                Only include keys that are relevant to the query.

                Query: {query_text}

                Parameters:
                """

                # Generate response
                response = self.local_llm(prompt, max_length=200, num_return_sequences=1)
                raw_result = response[0]['generated_text'].replace(prompt, "").strip()

                # Try to extract JSON
                try:
                    import json
                    # Look for JSON pattern in the response
                    json_match = re.search(r'({.*})', raw_result, re.DOTALL)
                    if json_match:
                        params = json.loads(json_match.group(1))
                        self.logger.info(f"Successfully extracted parameters using local LLM")
                        return params
                except Exception as e:
                    self.logger.warning(f"Failed to parse local LLM output as JSON: {e}")

            except Exception as e:
                self.logger.warning(f"Error using local LLM for parameter extraction: {e}")
                # Fall back to other methods

        if self.use_langchain:
            try:
                # Use LangChain for parameter extraction
                import json

                raw_result = self.param_chain.invoke({"query": query_text})

                # Try to extract JSON from markdown code blocks if present
                json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw_result)
                if json_match:
                    json_str = json_match.group(1).strip()
                else:
                    # If no code block, use the whole response
                    json_str = raw_result.strip()

                try:
                    # Parse the JSON result
                    params = json.loads(json_str)
                    return params
                except json.JSONDecodeError:
                    self.logger.warning(f"Could not parse LangChain parameter result as JSON: {raw_result}")
                    # Fall back to rule-based approach

            except Exception as e:
                self.logger.error(f"Error using LangChain for parameter extraction: {e}")
                # Fall back to rule-based approach

        # Rule-based parameter extraction
        parameters = {}

        # Extract model IDs
        model_ids = self._extract_model_mentions(query_text)
        if model_ids:
            parameters["model_ids"] = model_ids

        # Extract metrics of interest
        metrics = []
        for match in re.finditer(self.metric_pattern, query_text.lower()):
            metrics.append(match.group(1))
        if metrics:
            parameters["metrics"] = metrics

        # Extract filters
        filters = {}
        for filter_name, pattern in self.filter_patterns.items():
            for match in re.finditer(pattern, query_text.lower()):
                if filter_name == "architecture":
                    filters["architecture"] = match.group(1)
                elif filter_name == "framework":
                    filters["framework"] = match.group(1)
                elif filter_name == "params":
                    filters["params"] = {
                        "operator": match.group(2),
                        "value": match.group(3)
                    }
                elif filter_name == "date":
                    filters["date"] = {
                        "field": match.group(1),
                        "operator": match.group(2),
                        "value": match.group(3)
                    }

        if filters:
            parameters["filters"] = filters

        # Extract result limit
        limit_match = re.search(self.limit_pattern, query_text.lower())
        if limit_match:
            parameters["limit"] = int(limit_match.group(2))

        # Extract sorting criteria
        sort_match = re.search(self.sort_pattern, query_text.lower())
        if sort_match:
            parameters["sort_by"] = {
                "field": sort_match.group(3),
                "order": sort_match.group(4) if sort_match.group(4) else "descending"
            }

        # Add intent-specific parameter extraction
        if intent == QueryIntent.COMPARISON:
            parameters.update(self._extract_comparison_parameters(query_text))
        elif intent == QueryIntent.NOTEBOOK:
            parameters.update(self._extract_notebook_parameters(query_text))
        elif intent == QueryIntent.IMAGE_SEARCH:
            parameters.update(self._extract_image_parameters(query_text))

        return parameters

    def _extract_model_mentions(self, query_text: str) -> List[str]:
        """
        Extract mentions of model IDs or names from query text.

        Args:
            query_text: The query text to extract from

        Returns:
            List of model identifiers
        """
        model_ids = []

        # Try to extract explicit model_id mentions
        for match in re.finditer(self.model_id_pattern, query_text.lower()):
            model_ids.append(match.group(2))

        # Handle "X model" pattern (where X is the model name)
        model_suffix_pattern = r'(\w+)\s+model\b'
        for match in re.finditer(model_suffix_pattern, query_text.lower()):
            model_ids.append(match.group(1))

        # Check for model family mentions
        doc = self.nlp(query_text)
        for ent in doc.ents:
            if ent.label_ in ["ORG", "PRODUCT"]:
                model_ids.append(ent.text)

        # Check for common model family keywords
        for family in self.model_families:
            matches = re.finditer(r'\b' + re.escape(family) + r'[-_]?(\d+|v\d+)?\b', query_text.lower())
            for match in matches:
                model_ids.append(match.group(0))

        # Deduplicate and clean
        return list(set(model_ids))

    def _extract_comparison_parameters(self, query_text: str) -> Dict[str, Any]:
        """
        Extract comparison-specific parameters.

        Args:
            query_text: The query text to extract from

        Returns:
            Dictionary of comparison parameters
        """
        params = {}

        # Extract comparison dimensions
        dimensions = []
        dimension_pattern = r"(compare|comparing|comparison) (on|by|in terms of|regarding) ([\w\s,]+)"
        match = re.search(dimension_pattern, query_text.lower())
        if match:
            # Split by commas or 'and' and clean up
            dims = re.split(r',|\sand\s', match.group(3))
            dimensions = [dim.strip() for dim in dims if dim.strip()]

        if dimensions:
            params["comparison_dimensions"] = dimensions

        # Extract visualization preference
        visualization_pattern = r'(show|display|visualize|plot|graph|chart)'
        if re.search(visualization_pattern, query_text.lower()):
            params["visualize"] = True

        return params

    def _extract_notebook_parameters(self, query_text: str) -> Dict[str, Any]:
        """
        Extract notebook generation specific parameters.

        Args:
            query_text: The query text to extract from

        Returns:
            Dictionary of notebook parameters
        """
        params = {}

        # Extract analysis type
        analysis_types = []
        analysis_pattern = r"(analyze|analysis|examine|study|investigate) ([\w\s,]+)"
        match = re.search(analysis_pattern, query_text.lower())
        if match:
            # Split by commas or 'and' and clean up
            types = re.split(r',|\sand\s', match.group(2))
            analysis_types = [t.strip() for t in types if t.strip()]

        if analysis_types:
            params["analysis_types"] = analysis_types

        # Check for dataset mention
        dataset_pattern = r"(dataset|data)[:\s]+([\w\s-]+)"
        match = re.search(dataset_pattern, query_text.lower())
        if match:
            params["dataset"] = match.group(2).strip()

        # Check for resource constraints
        resource_pattern = r"(using|with) ([\w\s]+) (resources|gpu|memory|cpu)"
        match = re.search(resource_pattern, query_text.lower())
        if match:
            params["resources"] = match.group(2).strip()

        return params

    def _extract_image_parameters(self, query_text: str) -> Dict[str, Any]:
        """
        Extract image search specific parameters.

        Args:
            query_text: The query text to extract from

        Returns:
            Dictionary of image search parameters
        """
        params = {}

        # Extract prompt terms
        prompt_pattern = r"(prompt|prompts|text)[:\s]+[\"']?([\w\s,]+)[\"']?"
        match = re.search(prompt_pattern, query_text.lower())
        if match:
            params["prompt_terms"] = match.group(2).strip()

        # Extract style tags
        style_pattern = r"(style|type|category|look)[:\s]+[\"']?([\w\s,]+)[\"']?"
        match = re.search(style_pattern, query_text.lower())
        if match:
            # Split by commas and clean up
            styles = re.split(r',|\sand\s', match.group(2))
            params["style_tags"] = [style.strip() for style in styles if style.strip()]

        # Extract resolution preference
        resolution_pattern = r"(resolution|size|dimensions)[:\s]+(\d+)\s*[x×]\s*(\d+)"
        match = re.search(resolution_pattern, query_text.lower())
        if match:
            params["resolution"] = {
                "width": int(match.group(2)),
                "height": int(match.group(3))
            }

        # Flag to show model ID and image path
        params["show_model_id"] = True
        params["show_image_path"] = True

        return params

    def preprocess_query(self, query_text: str) -> str:
        """
        Preprocess a query for searching.

        Args:
            query_text: The raw query text

        Returns:
            Preprocessed query text
        """
        # Basic cleaning
        query_text = query_text.strip()

        # Use spaCy for tokenization and lemmatization
        doc = self.nlp(query_text)

        # Enhanced token handling
        processed_tokens = []
        skip_indices = set()

        # First pass: identify entities to preserve as-is
        for ent in doc.ents:
            if ent.label_ in ["PRODUCT", "ORG", "GPE", "PERSON", "WORK_OF_ART"]:
                for i in range(ent.start, ent.end):
                    skip_indices.add(i)
                processed_tokens.append(ent.text)

        # Second pass: handle noun phrases (potential model names)
        for chunk in doc.noun_chunks:
            # Check if any of the model families appear in this noun chunk
            contains_model = any(family in chunk.text.lower() for family in self.model_families)
            if contains_model:
                # Check if all tokens in this chunk are already skipped
                all_skipped = all(i in skip_indices for i in range(chunk.start, chunk.end))
                if not all_skipped:
                    # Add all tokens in this chunk to skip indices
                    for i in range(chunk.start, chunk.end):
                        skip_indices.add(i)
                    processed_tokens.append(chunk.text)

        # Third pass: process remaining tokens
        for i, token in enumerate(doc):
            if i not in skip_indices:
                # Skip stopwords, punctuation, and spaces
                if not token.is_stop and not token.is_punct and not token.is_space and len(token.text.strip()) > 1:
                    # Use lemma for standardization
                    processed_tokens.append(token.lemma_.lower())

        # Join all processed tokens
        return " ".join(processed_tokens)

    def get_intent_explanation(self, intent: QueryIntent, query_text: str) -> str:
        """
        Generate an explanation for why a particular intent was classified.

        Args:
            intent: The classified intent
            query_text: The original query text

        Returns:
            A human-readable explanation
        """
        explanation = f"I classified this as a {intent.value} query because "

        if intent == QueryIntent.RETRIEVAL:
            model_mentions = self._extract_model_mentions(query_text)
            if model_mentions:
                explanation += f"it mentions the model(s) {', '.join(model_mentions)}, "
                explanation += "and appears to be asking for information about them."
            else:
                explanation += "it uses retrieval-related terms and doesn't match other intent patterns."

        elif intent == QueryIntent.COMPARISON:
            model_mentions = self._extract_model_mentions(query_text)
            if len(model_mentions) > 1:
                explanation += f"it mentions multiple models ({', '.join(model_mentions)}) "
                explanation += "and uses comparison-related language."
            else:
                explanation += "it uses comparison terms like 'versus', 'better than', or 'difference between'."

        elif intent == QueryIntent.NOTEBOOK:
            explanation += "it requests the creation of code, a notebook, or an analysis script."

        elif intent == QueryIntent.IMAGE_SEARCH:
            explanation += "it asks for images, pictures, or visual examples generated by models."

        elif intent == QueryIntent.METADATA:
            explanation += "it's asking about the structure, fields, or properties of models rather than their function."

        else:  # UNKNOWN
            explanation += "it doesn't clearly match any of the known intent patterns."

        return explanation

    def format_result(self, parse_result: Dict[str, Any], include_explanation: bool = False) -> str:
        """
        Format the parse result into a human-readable string.

        Args:
            parse_result: The result from parse_query
            include_explanation: Whether to include an explanation of the classification

        Returns:
            A formatted string representation of the parse result
        """
        formatted = f"Intent: {parse_result['intent'].upper()}\n\n"

        if include_explanation:
            intent_enum = next(i for i in QueryIntent if i.value == parse_result['intent'])
            explanation = self.get_intent_explanation(intent_enum, parse_result.get('processed_query', ''))
            formatted += f"{explanation}\n\n"

        formatted += "Parameters:\n"

        params = parse_result['parameters']
        if not params:
            formatted += "  No specific parameters extracted.\n"
        else:
            for key, value in params.items():
                if key == 'model_ids' and value:
                    formatted += f"  Models: {', '.join(value)}\n"
                elif key == 'metrics' and value:
                    formatted += f"  Metrics: {', '.join(value)}\n"
                elif key == 'filters' and value:
                    formatted += "  Filters:\n"
                    for filter_key, filter_val in value.items():
                        if isinstance(filter_val, dict):
                            filter_desc = f"{filter_val.get('operator', '')} {filter_val.get('value', '')}"
                            formatted += f"    {filter_key}: {filter_desc}\n"
                        else:
                            formatted += f"    {filter_key}: {filter_val}\n"
                elif key == 'limit':
                    formatted += f"  Limit: {value}\n"
                elif key == 'sort_by' and value:
                    order = value.get('order', 'descending')
                    formatted += f"  Sort by: {value.get('field', '')} ({order})\n"
                elif key == 'comparison_dimensions' and value:
                    formatted += f"  Comparison dimensions: {', '.join(value)}\n"
                elif key == 'analysis_types' and value:
                    formatted += f"  Analysis types: {', '.join(value)}\n"
                elif key == 'dataset' and value:
                    formatted += f"  Dataset: {value}\n"
                elif key == 'prompt_terms' and value:
                    formatted += f"  Prompt terms: {value}\n"
                elif key == 'style_tags' and value:
                    formatted += f"  Style tags: {', '.join(value)}\n"
                elif key == 'resolution' and value:
                    formatted += f"  Resolution: {value.get('width', '')}x{value.get('height', '')}\n"

        return formatted