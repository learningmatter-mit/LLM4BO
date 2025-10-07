#!/usr/bin/env python3
"""Utility class for LLM API interactions."""

import os
import time
import random
import logging
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional, List, Callable
import re
import anthropic
import openai
from functools import wraps


def exponential_backoff_retry(max_attempts: int = 3, base_delay: float = 10.0, max_delay: float = 300.0):
    """Decorator for exponential backoff retry logic.
    
    Args:
        max_attempts: Maximum number of retry attempts (default: 3)
        base_delay: Base delay in seconds for first retry (default: 10.0)
        max_delay: Maximum delay in seconds (default: 300.0 = 5 minutes)
    
    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # Check if this is a retryable error
                    error_str = str(e).lower()
                    is_retryable = any(keyword in error_str for keyword in [
                        'overloaded', 'rate limit', 'rate_limit', 'too many requests',
                        'service unavailable', 'timeout', 'connection error',
                        'temporary failure', 'server error', '503', '429', '500', '502', '504'
                    ])
                    
                    if not is_retryable or attempt == max_attempts - 1:
                        # Either not retryable or last attempt - raise the error
                        raise e
                    
                    # Calculate delay: exponential backoff with jitter
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    jitter = random.uniform(0, min(delay * 0.1, 1.0))  # Up to 10% jitter, max 1s
                    total_delay = delay + jitter
                    
                    logger.warning(
                        f"API call failed with retryable error (attempt {attempt + 1}/{max_attempts}): {str(e)[:100]}..."
                    )
                    logger.info(f"Retrying in {total_delay:.1f}s")
                    time.sleep(total_delay)
                    
            # This should never be reached, but just in case
            raise Exception(f"Max retries ({max_attempts}) exceeded")
        return wrapper
    return decorator


class LLMAPI(ABC):
    """Abstract base class for LLM API interactions."""
    
    def __init__(self, **kwargs: Any) -> None:
        """Initialize LLM API client.
        
        Args:
            **kwargs: API-specific configuration parameters
        """
        self.hyperparameters = kwargs
        self.logger = logging.getLogger(self.__class__.__name__)
        self.client = self._initialize_client()
    
    @abstractmethod
    def _initialize_client(self):
        """Initialize the API client. To be implemented by subclasses."""
        raise NotImplementedError("Subclass must implement _initialize_client")
    
    @abstractmethod
    def _pass_to_llm_stream(self, prompt: str) -> Tuple[str, int]:
        """Send prompt to LLM and return response with token count. To be implemented by subclasses."""
        raise NotImplementedError("Subclass must implement _pass_to_llm_stream")
    
    @exponential_backoff_retry(max_attempts=3, base_delay=10.0, max_delay=300.0)
    def pass_to_llm(self, prompt: str) -> Tuple[str, int]:
        """Pass a prompt to the LLM and return the response.
        Uses exponential backoff retry with 3 attempts, starting at 10s, up to 5 min.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            Tuple of (response_text, input_tokens)
            
        Raises:
            Exception: If max retries exceeded
        """
        return self._pass_to_llm_stream(prompt)
    
    def parse_indicies(self, response: str, max_index: int, min_index: int) -> List[int]:
        """Parse integer indices from response text with robust error handling.
        
        Args:
            response: Text response containing indices
            max_index: Maximum valid index (inclusive)
            min_index: Minimum valid index (inclusive)
            
        Returns:
            List of unique, valid indices
            
        Raises:
            ValueError: If no valid indices found
        """
        def extract_indices(text: str) -> List[int]:
            """Try multiple extraction methods, preferring XML tags (AI should follow instructions)."""
            indices = []
            
            # Method 1 (preferred): XML-style tags <selected_indices>[list of indices]</selected_indices>, select last such instance if there are multiple
            tag_matches = re.findall(r'<selected_indices>\s*\[?([0-9,\s]+)\]?\s*</selected_indices>', text, re.IGNORECASE)
            if tag_matches:
                # Take the last match if multiple exist
                last_match = tag_matches[-1]
                indices = [int(x.strip()) for x in last_match.split(',') if x.strip()]
                return indices
            
            # AI didn't follow instructions - log warning and try fallbacks
            self.logger.warning("AI did not use expected XML tag format for indices")
            
            # Method 2 (fallback): Direct bracket notation [1,2,3] or [1, 2, 3]
            bracket_matches = re.findall(r'\[([0-9,\s]+)\]', text)
            if bracket_matches:
                # Take the last match if multiple exist
                last_bracket_match = bracket_matches[-1]
                indices = [int(x.strip()) for x in last_bracket_match.split(',') if x.strip()]
                self.logger.info("Successfully parsed indices using bracket notation fallback")
                return indices
            else:
                raise ValueError("No valid indices found in response")
                    
        def validate_indices(indices: List[int]) -> List[int]:
            """Remove duplicates and filter to valid range."""
            # Remove duplicates while preserving order
            # Filter to valid range
            valid_indices = [idx for idx in indices if min_index <= idx <= max_index]
            valid_indices = list(set(valid_indices))
            # Log warnings
            if len(valid_indices) != len(indices):
                self.logger.warning(f"Removed {len(indices) - len(valid_indices)} duplicate or out indices")
            
            return valid_indices
        
        try:
            # Extract indices using multiple methods
            raw_indices = extract_indices(response)
            
            if not raw_indices:
                raise ValueError("No numeric indices found in response")
            
            # Validate and clean indices
            valid_indices = validate_indices(raw_indices)
            
            if not valid_indices:
                raise ValueError(f"No valid indices found in range {min_index}-{max_index}")
            
            self.logger.info(f"Successfully parsed {len(valid_indices)} valid indices")
            self.logger.debug(f"Selected indices: {valid_indices}")
            
            return valid_indices
            
        except ValueError:
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error parsing indices: {str(e)}")
            self.logger.error(f"Response: {response[:200]}...")  # Truncate long responses
            raise ValueError(f"Failed to parse indices: {str(e)}")


class ClaudeAPI(LLMAPI):
    """Claude API implementation."""
    
    def __init__(self, **kwargs: Any) -> None:
        """Initialize Claude API client.
        
        Args:
            **kwargs: Claude-specific parameters
                - api_key: Claude API key (default: from environment or hardcoded)
                - model: Claude model name (default: claude-sonnet-4-20250514)
                - max_tokens: Maximum response tokens (default: 8000)
                - temperature: Sampling temperature (default: 0.3)
        """
        # Set default hyperparameters
        default_params = {
            'model': 'claude-sonnet-4-20250514',
            'max_tokens': 8000,
            'temperature': 0.3,
            'api_key': os.environ.get('ANTHROPIC_API_KEY')
        }
        
        # Update with provided kwargs
        for key, value in kwargs.items():
            default_params[key] = value
            
        super().__init__(**default_params)
    
    def _initialize_client(self):
        """Initialize Claude client."""
        return anthropic.Anthropic(
            api_key=self.hyperparameters['api_key']
        )
    
    def _pass_to_llm_stream(self, prompt: str) -> Tuple[str, int]:
        """Send prompt to Claude and return response with token count.
        
        Args:
            prompt: The prompt to send to Claude
            
        Returns:
            Tuple of (response_text, input_tokens)
        """
        full_response = ""
        input_tokens = 0
        self.logger.info(f"Sending prompt to {self.hyperparameters['model']} with temperature {self.hyperparameters['temperature']}")
        
        try:
            with self.client.messages.stream(
                model=self.hyperparameters['model'],
                max_tokens=self.hyperparameters['max_tokens'],
                messages=[{"role": "user", "content": prompt}],
                temperature=self.hyperparameters['temperature']
            ) as stream:
                for event in stream:
                    if event.type == 'content_block_delta':
                        if hasattr(event.delta, 'text'):
                            full_response += event.delta.text
                            self.logger.debug(f"Received chunk: {event.delta.text}")
                    if event.type == 'message_start':
                        input_tokens += event.message.usage.input_tokens
        except Exception as e:
            # Convert Anthropic-specific errors to more generic format for retry logic
            if hasattr(e, 'status_code'):
                if e.status_code in [429, 503, 500, 502, 504]:
                    raise Exception(f"API Error {e.status_code}: {str(e)}")
            raise e
                    
        return full_response, input_tokens


class LambdaAPI(LLMAPI):
    """OpenAI API implementation."""
    
    def __init__(self, **kwargs: Any) -> None:
        """Initialize OpenAI API client.
        
        Args:
            **kwargs: OpenAI-specific parameters
                - api_key: OpenAI API key (default: from environment)
                - base_url: Base URL for API (default: None for OpenAI, or custom endpoint)
                - model: Model name (default: None, must be specified)
                - max_tokens: Maximum response tokens (default: 4000)
                - temperature: Sampling temperature (default: 0.3)
        """
        # Set default hyperparameters
        default_params = {
            'api_key': os.getenv("LAMBDA_API_KEY"),
            'base_url': "https://api.lambda.ai/v1",
            'model': 'llama-4-maverick-17b-128e-instruct-fp8',
            'max_tokens': 32000,
            'temperature': 0.6
        }
        
        # Update with provided kwargs
        for key, value in kwargs.items():
            default_params[key] = value
        super().__init__(**default_params)
    
    def _initialize_client(self):
        """Initialize OpenAI client."""
        client_kwargs = {
            'api_key': self.hyperparameters['api_key']
        }
        
        if self.hyperparameters['base_url']:
            client_kwargs['base_url'] = self.hyperparameters['base_url']
            
        return openai.OpenAI(**client_kwargs)
    
    def _pass_to_llm_stream(self, prompt: str) -> Tuple[str, int]:
        """Send prompt to OpenAI model and return response with token count.
        
        Args:
            prompt: The prompt to send to the model
            
        Returns:
            Tuple of (response_text, total_tokens)
        """
        if not self.hyperparameters['model']:
            raise ValueError("Model name must be specified for OpenAI API")
        self.logger.info(f"Sending prompt to {self.hyperparameters['model']} with temperature {self.hyperparameters['temperature']}")
        
        try:
            response = self.client.chat.completions.create(
                model=self.hyperparameters['model'],
                messages=[{"role": "system", "content": "You are a chemoinformatic expert."}, {"role": "user", "content": prompt}],
                temperature=self.hyperparameters['temperature'],
                extra_body={
                    "repetition_penalty": 1.1
                }
                )
        except Exception as e:
            # Convert OpenAI-specific errors to more generic format for retry logic
            if hasattr(e, 'status_code'):
                if e.status_code in [429, 503, 500, 502, 504]:
                    raise Exception(f"API Error {e.status_code}: {str(e)}")
            raise e
        print(response)
        message = response.choices[0].message.content
        total_tokens = response.usage.total_tokens
        
        return message, total_tokens


class OpenAIAPI(LLMAPI):
    """OpenAI API implementation."""
    
    def __init__(self, **kwargs: Any) -> None:
        """Initialize OpenAI API client.
        
        Args:
            **kwargs: OpenAI-specific parameters
                - api_key: OpenAI API key (default: from environment)
                - base_url: Base URL for API (default: None for OpenAI, or custom endpoint)
                - model: Model name (default: None, must be specified)
                - max_tokens: Maximum response tokens (default: 4000)
                - temperature: Sampling temperature (default: 0.3)
        """
        # Set default hyperparameters
        self.default_params = {
            'api_key': os.getenv("OPENAI_API_KEY"),
            'model': 'gpt-5',
            'reasoning_effort': 'medium',
            'max_completion_tokens': 20000
        }
        
        # Update with provided kwargs
        for key, value in kwargs.items():
            self.default_params[key] = value
        super().__init__(**self.default_params)
    
    def _initialize_client(self):
        """Initialize OpenAI client."""
        client_kwargs = {
            'api_key': self.default_params['api_key']
        }
        return openai.OpenAI(**client_kwargs)
    
    def _pass_to_llm_stream(self, prompt: str) -> Tuple[str, int]:
        """Send prompt to OpenAI model and return response with token count.
        
        Args:
            prompt: The prompt to send to the model
            
        Returns:
            Tuple of (response_text, total_tokens)
        """
        if not self.hyperparameters['model']:
            raise ValueError("Model name must be specified for OpenAI API")
        self.logger.info(f"Sending prompt to {self.hyperparameters['model']}")
        if "/think" not in prompt:
            self.default_params['reasoning_effort'] = 'low'
            self.default_params['max_completion_tokens'] = 10000

        try:
            response = self.client.chat.completions.create(
                model=self.hyperparameters['model'],
                messages=[{"role": "user", "content": prompt}],
                reasoning_effort=self.default_params['reasoning_effort'],
                max_completion_tokens=self.default_params['max_completion_tokens']
                )
        except Exception as e:
            # Convert OpenAI-specific errors to more generic format for retry logic
            if hasattr(e, 'status_code'):
                if e.status_code in [429, 503, 500, 502, 504]:
                    raise Exception(f"API Error {e.status_code}: {str(e)}")
            raise e
            
        print(response)
        message = response.choices[0].message.content
        total_tokens = response.usage.total_tokens
        
        return message, total_tokens


def with_retry(func: Callable, *args, **kwargs):
    """Apply exponential backoff retry to any function call.
    
    Args:
        func: Function to call with retry logic
        *args: Positional arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        Result of the function call
    """
    @exponential_backoff_retry(max_attempts=3, base_delay=10.0, max_delay=300.0)
    def wrapper():
        return func(*args, **kwargs)
    
    return wrapper()
