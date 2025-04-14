# src/response_generator/llm_interface.py
import json
import logging
import time
from typing import Dict, Any, List, Union

import requests


class LLMInterface:
    """
    Interface for communicating with Language Models, focused on Ollama.
    """
    
    def __init__(self, 
                 provider="ollama", 
                 model_name="mistral:latest",
                 base_url="http://localhost:11434",
                 timeout=120,
                 max_retries=3,
                 retry_delay=2):
        """
        Initialize the LLM interface.
        
        Args:
            provider: LLM provider (currently supports 'ollama')
            model_name: Name of the model to use
            base_url: Base URL for the Ollama API
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            retry_delay: Delay between retries in seconds
        """
        self.provider = provider
        self.model_name = model_name
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.logger = logging.getLogger(__name__)
        
        # Initialize the client
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the LLM client based on the provider."""
        if self.provider == "ollama":
            # Check if Ollama is available and if the model is available
            try:
                response = requests.get(f"{self.base_url}/api/tags", timeout=self.timeout)
                response.raise_for_status()
                
                models = response.json().get("models", [])
                model_names = [model.get("name") for model in models]
                
                if self.model_name not in model_names:
                    self.logger.warning(f"Model {self.model_name} not found in Ollama. Available models: {model_names}")
                    self.logger.info(f"Attempting to pull model {self.model_name}")
                    pull_response = requests.post(
                        f"{self.base_url}/api/pull",
                        json={"name": self.model_name},
                        timeout=self.timeout
                    )
                    pull_response.raise_for_status()
                    self.logger.info(f"Successfully pulled model {self.model_name}")
                
                self.logger.info(f"Initialized Ollama client with model {self.model_name}")
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Failed to initialize Ollama client: {str(e)}")
                raise RuntimeError(f"Failed to initialize Ollama client: {str(e)}")
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def generate_response(self, 
                          prompt: str, 
                          temperature: float = 0.7, 
                          max_tokens: int = 1000,
                          streaming: bool = False) -> Union[str, List[str]]:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            temperature: Temperature for sampling (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            streaming: Whether to stream the response
            
        Returns:
            Generated response as a string, or a list of response chunks if streaming
        """
        return self.handle_rate_limiting(
            self._generate_response,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming
        )

    def _generate_response(self,
                           prompt: str,
                           temperature: float = 0.7,
                           max_tokens: int = 1000,
                           streaming: bool = False) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Internal implementation of generate_response with provider-specific logic,
           returning a structured dictionary instead of a plain string."""
        if self.provider == "ollama":
            endpoint = f"{self.base_url}/api/{'generate' if not streaming else 'chat'}"

            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "temperature": temperature,
                "num_predict": max_tokens,
                "stream": streaming
            }

            if streaming:
                # For the chat endpoint, format is slightly different:
                payload = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "stream": True
                }
                # Get the list of chunks and then join them
                chunks = self._stream_response(endpoint, payload)
                full_response = "".join(chunks)
                return {"id": "llm_response_0", "content": full_response, "metadata": {}}
            else:
                response = requests.post(endpoint, json=payload, timeout=self.timeout)
                response.raise_for_status()
                generated_text = response.json().get("response", "")
                return {"id": "llm_response_0", "content": generated_text, "metadata": {}}
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _stream_response(self, endpoint: str, payload: Dict[str, Any]) -> List[str]:
        """
        Stream a response from the LLM.
        
        Args:
            endpoint: API endpoint
            payload: Request payload
            
        Returns:
            List of response chunks
        """
        chunks = []
        response = requests.post(endpoint, json=payload, stream=True, timeout=self.timeout)
        
        for line in response.iter_lines():
            if line:
                try:
                    chunk_data = json.loads(line)
                    if 'response' in chunk_data:
                        chunks.append(chunk_data['response'])
                    
                    # Check if the response is complete
                    if chunk_data.get('done', False):
                        break
                except json.JSONDecodeError:
                    self.logger.warning(f"Failed to decode JSON from line: {line}")
        
        return chunks
    
    def generate_structured_response(self, 
                                    system_prompt: str, 
                                    user_prompt: str, 
                                    temperature: float = 0.7, 
                                    max_tokens: int = 1000) -> str:
        """
        Generate a structured response from the LLM using a system prompt and user prompt.
        
        Args:
            system_prompt: System instructions for the LLM
            user_prompt: User query or content
            temperature: Temperature for sampling (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated response
        """
        return self.handle_rate_limiting(
            self._generate_structured_response,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )

    def _generate_structured_response(self,
                                      system_prompt: str,
                                      user_prompt: str,
                                      temperature: float = 0.7,
                                      max_tokens: int = 1000) -> Dict[str, Any]:
        """Internal implementation of generate_structured_response with provider-specific logic,
           returning a structured dictionary."""
        if self.provider == "ollama":
            endpoint = f"{self.base_url}/api/chat"

            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": temperature,
                # Note: Ollama's chat API doesn't have a direct max_tokens parameter
                "options": {
                    "num_predict": max_tokens
                }
            }

            response = requests.post(endpoint, json=payload, timeout=self.timeout)
            response.raise_for_status()

            try:
                content = response.json().get("message", {}).get("content", "")
                return {"id": "llm_structured_response_0", "content": content, "metadata": {}}
            except (KeyError, json.JSONDecodeError):
                self.logger.error(f"Failed to parse response: {response.text}")
                return {"id": "llm_structured_response_0", "content": "", "metadata": {}}
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

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
                if "429" in str(e) or "rate limit" in str(e).lower():
                    retries += 1
                    if retries > self.max_retries:
                        self.logger.error(f"Max retries ({self.max_retries}) exceeded due to rate limiting")
                        raise
                    
                    # Exponential backoff
                    sleep_time = self.retry_delay * (2 ** (retries - 1))
                    self.logger.warning(f"Rate limited. Retrying in {sleep_time} seconds. (Attempt {retries}/{self.max_retries})")
                    time.sleep(sleep_time)
                else:
                    # Re-raise if it's not a rate limit error
                    self.logger.error(f"Request failed: {str(e)}")
                    raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the currently loaded model.
        
        Returns:
            Dictionary with model information
        """
        if self.provider == "ollama":
            try:
                response = requests.get(f"{self.base_url}/api/show?name={self.model_name}", timeout=self.timeout)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Failed to get model info: {str(e)}")
                return {}
        else:
            return {}
    
    def list_available_models(self) -> List[str]:
        """
        List available models from the provider.
        
        Returns:
            List of available model names
        """
        if self.provider == "ollama":
            try:
                response = requests.get(f"{self.base_url}/api/tags", timeout=self.timeout)
                response.raise_for_status()
                models = response.json().get("models", [])
                return [model.get("name") for model in models]
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Failed to list available models: {str(e)}")
                return []
        else:
            return []

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize the interface
    llm = LLMInterface(provider="ollama", model_name="mistral:latest")
    
    # Generate a simple response
    response = llm.generate_response(
        prompt="Explain what a RAG system is in 3 sentences.",
        temperature=0.7,
        max_tokens=200
    )
    
    print("Response:", response)
    
    # Generate a structured response
    structured_response = llm.generate_structured_response(
        system_prompt="You are an AI assistant that provides concise answers.",
        user_prompt="What is the capital of France?",
        temperature=0.5,
        max_tokens=100
    )
    
    print("Structured Response:", structured_response)
