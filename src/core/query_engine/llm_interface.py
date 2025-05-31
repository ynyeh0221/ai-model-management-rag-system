import json
import logging
import time
from typing import Dict, Any, List, Union

import requests


class LLMInterface:
    """
    Interface for communicating with Language Models, focused on Ollama with Apple Silicon GPU support.
    """

    def __init__(
        self,
        provider: str = "ollama",
        model_name: str = "deepseek-r1:7b",
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
        max_retries: int = 3,
        retry_delay: int = 2,
        use_gpu: bool = True,
        gpu_layers: Union[int, None] = None,
    ):
        """
        Initialize the LLM interface.

        Args:
            provider: LLM provider (currently supports 'ollama')
            model_name: Name of the model to use
            base_url: Base URL for the Ollama API
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            retry_delay: Delay between retries in seconds
            use_gpu: Whether to use GPU acceleration (Metal on Mac)
            gpu_layers: Number of layers to offload to GPU (None = all layers)
        """
        self.provider = provider
        self.model_name = model_name
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.use_gpu = use_gpu
        self.gpu_layers = gpu_layers
        self.logger = logging.getLogger(__name__)

        self._initialize_client()

    # ─── Public methods ─────────────────────────────────────────────────────────

    def generate_response(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        streaming: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate a response from the LLM.

        Args:
            prompt: The prompt to send to the LLM
            temperature: Temperature for sampling (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            streaming: Whether to stream the response

        Returns:
            A dictionary with:
              - id: "llm_response_0"
              - content: <generated text or concatenated chunks>
              - metadata: {}
        """
        return self.handle_rate_limiting(
            self._generate_response, prompt=prompt, temperature=temperature, max_tokens=max_tokens, streaming=streaming
        )

    def generate_structured_response(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> Dict[str, Any]:
        """
        Generate a structured response from the LLM using a system prompt and user prompt.

        Args:
            system_prompt: System instructions for the LLM
            user_prompt: User query or content
            temperature: Temperature for sampling (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate

        Returns:
            A dictionary with:
              - id: "llm_structured_response_0"
              - content: <generated structured text>
              - metadata: {}
        """
        return self.handle_rate_limiting(
            self._generate_structured_response,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def list_available_models(self) -> List[str]:
        """
        List available models from the provider.

        Returns:
            List of available model names
        """
        if self.provider != "ollama":
            return []

        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            response.raise_for_status()
            models = response.json().get("models", [])
            return [model.get("name") for model in models]
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to list available models: {e}")
            return []

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the currently loaded model.

        Returns:
            Dictionary with model information
        """
        if self.provider != "ollama":
            return {}

        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            response.raise_for_status()
            models = response.json().get("models", [])
            for model in models:
                if model.get("name") == self.model_name:
                    return model
            self.logger.warning(f"Model '{self.model_name}' not found in model list.")
            return {}
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to get model info: {e}")
            return {}

    # ─── Internal methods ────────────────────────────────────────────────────────

    def _initialize_client(self) -> None:
        """
        Initialize the LLM client based on the provider.
        This method has been broken into smaller helpers to reduce cognitive complexity.
        """
        if self.provider != "ollama":
            raise ValueError(f"Unsupported provider: {self.provider}")

        try:
            tags_resp = requests.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            tags_resp.raise_for_status()
            available_models = self._extract_model_names(tags_resp.json())

            if self.model_name not in available_models:
                self.logger.warning(
                    f"Model '{self.model_name}' not found. Available models: {available_models}"
                )
                self._pull_model_with_gpu_if_needed()

            self.logger.info(f"Initialized Ollama client with model '{self.model_name}'")
            if self.use_gpu:
                self._log_gpu_info()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to initialize Ollama client: {e}")
            raise RuntimeError(f"Failed to initialize Ollama client: {e}")

    def _extract_model_names(self, tags_payload: Dict[str, Any]) -> List[str]:
        """
        From the /api/tags response, extract a list of model names.
        """
        models = tags_payload.get("models", [])
        return [model.get("name") for model in models]

    def _pull_model_with_gpu_if_needed(self) -> None:
        """
        Attempt to pull the model from Ollama. If GPU is enabled, add GPU options.
        """
        pull_payload: Dict[str, Any] = {"name": self.model_name}

        if self.use_gpu:
            options: Dict[str, Any] = {}
            if self.gpu_layers is not None:
                options["gpu_layers"] = self.gpu_layers
            else:
                # Default: use all layers on GPU
                options["gpu_layers"] = 99
            pull_payload["options"] = options
            self.logger.info(f"Pulling model with GPU acceleration. Options: {options}")

        pull_resp = requests.post(
            f"{self.base_url}/api/pull", json=pull_payload, timeout=self.timeout
        )
        pull_resp.raise_for_status()
        self.logger.info(f"Successfully pulled model '{self.model_name}'")

    def _log_gpu_info(self) -> None:
        """
        Retrieve model info and log GPU-related details.
        """
        model_info = self.get_model_info()
        params = model_info.get("parameters", {})
        self.logger.info(f"Model parameters: {params}")

        model_info_str = json.dumps(model_info).lower()
        if "metal" in model_info_str:
            self.logger.info("Metal GPU acceleration is enabled")
        else:
            self.logger.warning("Metal GPU acceleration may not be enabled")

    def _generate_response(
        self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000, streaming: bool = False
    ) -> Dict[str, Any]:
        """
        Internal implementation of generate_response with provider-specific logic.
        - Returns a single dict {"id": "llm_response_0", "content": "<...>", "metadata": {}}
          regardless of streaming or non‐streaming.
        """
        if self.provider != "ollama":
            raise ValueError(f"Unsupported provider: {self.provider}")

        endpoint = self._choose_endpoint(streaming)
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "temperature": temperature,
        }

        # Build GPU options (if any) in a separate statement
        if self.use_gpu:
            options: Dict[str, Any] = {"gpu_layers": self.gpu_layers if self.gpu_layers is not None else 99}
            payload["options"] = options

        if streaming:
            # Collect all chunks as strings, then concatenate
            chunks = self._collect_stream_chunks(endpoint, prompt, payload)
            full_content = "".join(chunks)
            return {"id": "llm_response_0", "content": full_content, "metadata": {}}

        # Non‐streaming: single request → single response
        payload.update({"prompt": prompt, "num_predict": max_tokens, "stream": False})

        # Ensure we have an 'options' key so we can override gpu_layers if needed
        if "options" not in payload:
            payload["options"] = {}
        payload["options"]["gpu_layers"] = self.gpu_layers if self.gpu_layers is not None else 99

        response = requests.post(endpoint, json=payload, timeout=self.timeout)
        response.raise_for_status()

        generated_text = response.json().get("response", "")
        return {"id": "llm_response_0", "content": generated_text, "metadata": {}}

    def _choose_endpoint(self, streaming: bool) -> str:
        """
        Decide which Ollama endpoint to hit based on the streaming flag.
        """
        if streaming:
            return f"{self.base_url}/api/chat"
        return f"{self.base_url}/api/generate"

    def _collect_stream_chunks(
        self, endpoint: str, prompt: str, payload: Dict[str, Any]
    ) -> List[str]:
        """
        Send a streaming request, collect each chunk's "response" text, and return them as a list.
        """
        payload.update(
            {
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
            }
        )
        response = requests.post(endpoint, json=payload, stream=True, timeout=self.timeout)
        response.raise_for_status()

        chunks: List[str] = []
        for line in response.iter_lines():
            if not line:
                continue
            try:
                chunk_data = json.loads(line)
            except json.JSONDecodeError:
                self.logger.warning(f"Failed to decode JSON from line: {line}")
                continue

            if "response" in chunk_data:
                chunks.append(chunk_data["response"])
            if chunk_data.get("done", False):
                break

        return chunks

    def _generate_structured_response(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> Dict[str, Any]:
        """
        Internal implementation of generate_structured_response with provider-specific logic,
        returning a structured dictionary.
        """
        if self.provider != "ollama":
            raise ValueError(f"Unsupported provider: {self.provider}")

        endpoint = f"{self.base_url}/api/chat"
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "stream": False,
        }

        # Extracted conditional into separate statements for clarity:
        if self.use_gpu:
            options = {"num_predict": max_tokens, "gpu_layers": self.gpu_layers if self.gpu_layers is not None else 99}
        else:
            options = {"num_predict": max_tokens}
        payload["options"] = options

        response = requests.post(endpoint, json=payload, timeout=self.timeout)
        response.raise_for_status()

        try:
            content = response.json().get("message", {}).get("content", "")
        except (KeyError, json.JSONDecodeError):
            self.logger.error(f"Failed to parse response: {response.text}")
            content = ""

        return {"id": "llm_structured_response_0", "content": content, "metadata": {}}

    def handle_rate_limiting(self, func, *args, **kwargs):
        """
        Handle rate limiting by retrying with exponential backoff.

        Args:
            func: Function to execute
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Returns:
            Result of the function
        """
        retries = 0
        while retries <= self.max_retries:
            try:
                return func(*args, **kwargs)
            except requests.exceptions.RequestException as e:
                error_str = str(e).lower()
                if "429" in error_str or "rate limit" in error_str:
                    retries += 1
                    if retries > self.max_retries:
                        self.logger.error(f"Max retries ({self.max_retries}) exceeded due to rate limiting")
                        raise
                    sleep_time = self.retry_delay * (2 ** (retries - 1))
                    self.logger.warning(
                        f"Rate limited. Retrying in {sleep_time} seconds. (Attempt {retries}/{self.max_retries})"
                    )
                    time.sleep(sleep_time)
                else:
                    self.logger.error(f"Request failed: {e}")
                    raise


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Initialize the interface with Apple Silicon GPU support
    llm = LLMInterface(
        provider="ollama", model_name="deepseek-r1:7b", use_gpu=True, gpu_layers=None
    )

    # Generate a simple response
    response = llm.generate_response(
        prompt="Explain what a RAG system is in 3 sentences.",
        temperature=0.7,
        max_tokens=200,
    )
    print("Response:", response)

    # Generate a structured response
    structured_response = llm.generate_structured_response(
        system_prompt="You are an AI assistant that provides concise answers.",
        user_prompt="What is the capital of France?",
        temperature=0.5,
        max_tokens=100,
    )
    print("Structured Response:", structured_response)

    # Check GPU acceleration status
    print("\nChecking GPU acceleration status:")
    model_info = llm.get_model_info()

    if model_info:
        model_name = model_info.get("name", "unknown")
        details = model_info.get("details", {})
        print(f"Model name: {model_name}")
        print("Model details:")
        for k, v in details.items():
            print(f"  - {k}: {v}")

        gpu_layers = details.get("gpu_layers", None)
        if gpu_layers:
            print(f"Configured to use GPU (gpu_layers = {gpu_layers}). Metal acceleration is likely active.")
        else:
            print("GPU layers not configured in metadata. Metal acceleration may still be active if passed during generation.")
    else:
        print("Could not fetch model info.")
