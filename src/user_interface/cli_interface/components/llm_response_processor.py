import json


class LLMResponseProcessor:
    """Processes and displays LLM responses."""

    @staticmethod
    def print_llm_content(response):
        """
        Extract and print content from an LLM response in various formats.

        Args:
            response: The response from the LLM, which could be a string, dict, or list.
        """
        try:
            # Handle dict case
            if isinstance(response, dict):
                LLMResponseProcessor._print_dict_response(response)
            # Handle str case
            elif isinstance(response, str):
                LLMResponseProcessor._print_str_response(response)
            # Handle list case
            elif isinstance(response, list):
                LLMResponseProcessor._print_list_response(response)
            else:
                print(f"Unsupported response type: {type(response)}")
                # Try to print it anyway
                print(str(response))

        except Exception as e:
            print(f"Error extracting content: {e}")
            # Try to print the raw response
            try:
                print("Raw response:")
                print(str(response)[:500])  # Print first 500 chars at most
            except:
                print("Could not print raw response")

    @staticmethod
    def _print_dict_response(response):
        """Handle dictionary response formats."""
        if "content" in response:
            print(response["content"])
        elif "text" in response:
            print(response["text"])
        elif "response" in response:
            print(response["response"])
        elif "answer" in response:
            print(response["answer"])
        elif "message" in response:
            # Check for OpenAI format
            if isinstance(response["message"], dict) and "content" in response["message"]:
                print(response["message"]["content"])
            else:
                print(response["message"])
        else:
            print("No recognizable content field found in the dictionary response")
            print(f"Available fields: {list(response.keys())}")

            # Try to print the full dictionary if it's not too large
            if len(str(response)) < 1000:
                print("Response content:")
                print(response)

    @staticmethod
    def _print_str_response(response):
        """Handle string response formats."""
        # Try to parse as JSON first
        try:
            parsed = json.loads(response)

            if isinstance(parsed, dict):
                # Recursively call with the parsed dict
                LLMResponseProcessor._print_dict_response(parsed)
            else:
                # Just print the string as is
                print(response)
        except json.JSONDecodeError:
            # Not JSON, print as is
            print(response)

    @staticmethod
    def _print_list_response(response):
        """Handle list response formats."""
        if response and len(response) > 0:
            # Try the first element
            first_elem = response[0]
            if isinstance(first_elem, dict):
                # Recursively call with the first dict
                LLMResponseProcessor._print_dict_response(first_elem)
            else:
                print(first_elem)
        else:
            print("Empty list response")