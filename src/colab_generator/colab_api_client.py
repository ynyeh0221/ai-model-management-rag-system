"""
Colab API Client for the AI Model Management RAG System.

This module provides functionality to create, execute, and manage Colab notebooks
through the Google API, supporting the notebook generation workflow described
in the system design document.
"""

import os
import json
import time
import logging
import tempfile
from typing import Dict, Optional, Any, List

# Google API libraries
from google.oauth2.service_account import Credentials
from google.oauth2 import credentials
from google.auth.transport.requests import Request
from google.auth.exceptions import RefreshError
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from googleapiclient.errors import HttpError

# Notebook libraries
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

class AuthenticationError(Exception): pass
class NotebookCreationError(Exception): pass
class ExecutionError(Exception): pass
class ResourceExceededError(Exception): pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
SCOPES = [
    'https://www.googleapis.com/auth/drive',
    'https://www.googleapis.com/auth/drive.file',
    'https://www.googleapis.com/auth/drive.appdata',
    'https://www.googleapis.com/auth/colab',
]
DEFAULT_TIMEOUT = 300  # 5 minutes
MAX_EXECUTION_WAIT_TIME = 7200  # 2 hours
POLL_INTERVAL = 10  # 10 seconds
TOKEN_PATH = 'token.json'

class ColabAPIClient:
    """
    Client for interacting with Google Colab and Drive APIs to create, 
    execute, and manage notebooks.
    
    This class handles authentication with Google APIs, notebook creation,
    execution monitoring, and result retrieval, supporting the notebook
    generation workflow described in the system design document.
    """
    
    def __init__(self, credentials_path: Optional[str] = None, test_mode: bool = True):
        """
        Initialize the Colab API Client.
        
        Args:
            credentials_path: Path to the Google API credentials JSON file.
                If None, will look for credentials in environment variables
                or default locations.
        """
        self.credentials_path = credentials_path
        self.drive_service = None
        self.colab_service = None

        # Skip authentication in test mode
        self._test_mode = test_mode
        if self._test_mode:
            logger.info("ColabAPIClient initialized in test mode")
        else:
            logger.info("ColabAPIClient initialized")

        self._authenticate()
    
    def _authenticate(self):
        """
        Authenticate with Google APIs using service account or OAuth2.
        
        This method attempts to authenticate using:
        1. Service account credentials if provided
        2. Existing OAuth token if available
        3. OAuth2 flow if needed
        
        Raises:
            AuthenticationError: If authentication fails
        """
        if not self._test_mode:
            try:
                creds = None

                # First check for service account credentials
                if self.credentials_path and os.path.exists(self.credentials_path):
                    try:
                        if self.credentials_path.endswith('.json'):
                            with open(self.credentials_path, 'r') as f:
                                creds_data = json.load(f)

                            # Check if this is a service account
                            if 'type' in creds_data and creds_data['type'] == 'service_account':
                                creds = Credentials.from_service_account_file(
                                    self.credentials_path, scopes=SCOPES
                                )
                                logger.info("Authenticated using service account")
                    except Exception as e:
                        logger.warning(f"Failed to load service account credentials: {e}")

                # If no service account, try OAuth
                if not creds:
                    # Check for existing token
                    if os.path.exists(TOKEN_PATH):
                        with open(TOKEN_PATH, 'r') as token:
                            creds = credentials.Credentials.from_authorized_user_info(
                                json.load(token), SCOPES
                            )

                    # If credentials expired, refresh them
                    if creds and creds.expired and creds.refresh_token:
                        creds.refresh(Request())
                        logger.info("Refreshed expired OAuth credentials")

                    # If no valid credentials, run OAuth flow
                    if not creds or not creds.valid:
                        if not self.credentials_path:
                            raise AuthenticationError(
                                "No credentials file provided for OAuth flow"
                            )

                        flow = InstalledAppFlow.from_client_secrets_file(
                            self.credentials_path, SCOPES
                        )
                        creds = flow.run_local_server(port=0)

                        # Save the credentials for future use
                        with open(TOKEN_PATH, 'w') as token:
                            token.write(creds.to_json())
                        logger.info("New OAuth credentials obtained and saved")

                # Build the Google API services
                self.drive_service = build('drive', 'v3', credentials=creds)
                self.colab_service = build('colab', 'v1', credentials=creds)
                logger.info("Google API services initialized successfully")

            except (RefreshError, FileNotFoundError, HttpError) as e:
                message = f"Authentication failed: {str(e)}"
                logger.error(message)
                raise AuthenticationError(message) from e
        else:
            logger.info("Authentication skipped (test mode)")
    
    def create_notebook(
        self, 
        notebook_content: Any, 
        filename: str, 
        folder_id: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new notebook in Google Drive.
        
        Args:
            notebook_content: The notebook content, either as an nbformat.NotebookNode
                             or a dict that can be converted to a notebook.
            filename: Name of the notebook file to create.
            folder_id: ID of the Google Drive folder to put the notebook in.
                      If None, the notebook will be created in the root of My Drive.
            description: Optional description for the file.
            metadata: Optional metadata to attach to the file.
            
        Returns:
            str: The ID of the created notebook file in Google Drive.
            
        Raises:
            NotebookCreationError: If notebook creation fails.
        """
        if not self.drive_service:
            self._authenticate()
        
        try:
            # Ensure the filename ends with .ipynb
            if not filename.endswith('.ipynb'):
                filename += '.ipynb'
            
            # Convert notebook content to nbformat if it's a dict
            if isinstance(notebook_content, dict):
                notebook = nbformat.from_dict(notebook_content)
            elif not isinstance(notebook_content, nbformat.NotebookNode):
                # If it's just raw content, create a new notebook
                notebook = new_notebook()
                
                # If it's a string, assume it's markdown
                if isinstance(notebook_content, str):
                    notebook.cells.append(new_markdown_cell(notebook_content))
                else:
                    logger.warning("Notebook content type not recognized, creating empty notebook")
            else:
                notebook = notebook_content
            
            # Add custom metadata if provided
            if metadata:
                if not hasattr(notebook, 'metadata'):
                    notebook.metadata = {}
                notebook.metadata.update(metadata)
            
            # Create a temporary file to upload
            with tempfile.NamedTemporaryFile(suffix='.ipynb', delete=False) as temp_file:
                temp_path = temp_file.name
                nbformat.write(notebook, temp_file)
            
            # File metadata for Google Drive
            file_metadata = {
                'name': filename,
                'mimeType': 'application/x-ipynb+json',
            }
            
            if description:
                file_metadata['description'] = description
                
            if folder_id:
                file_metadata['parents'] = [folder_id]
            
            # Upload the file
            media = MediaFileUpload(
                temp_path,
                mimetype='application/x-ipynb+json',
                resumable=True
            )
            
            file = self.drive_service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()
            
            # Clean up the temporary file
            os.unlink(temp_path)
            
            file_id = file.get('id')
            logger.info(f"Created notebook with ID: {file_id}")
            return file_id
            
        except Exception as e:
            message = f"Failed to create notebook: {str(e)}"
            logger.error(message)
            raise NotebookCreationError(message) from e
    
    def execute_notebook(
        self, 
        file_id: str, 
        parameters: Optional[Dict[str, Any]] = None,
        timeout: int = DEFAULT_TIMEOUT,
        machine_type: str = "STANDARD",
        max_execution_duration: int = 3600,  # 1 hour
        accelerator_type: Optional[str] = None,
        accelerator_count: int = 0
    ) -> str:
        """
        Execute a notebook in Colab.
        
        Args:
            file_id: The ID of the notebook file in Google Drive.
            parameters: Optional parameters to pass to the notebook execution.
            timeout: Timeout in seconds for the execution request (not the execution itself).
            machine_type: Type of machine to use for execution.
                        Options: "STANDARD", "HIGH_MEM", "HIGH_CPU"
            max_execution_duration: Maximum duration for notebook execution in seconds.
            accelerator_type: Type of accelerator to use, e.g., "GPU" or "TPU".
            accelerator_count: Number of accelerators to use.
            
        Returns:
            str: The execution ID of the running notebook.
            
        Raises:
            ExecutionError: If notebook execution fails.
        """
        if not self.colab_service:
            self._authenticate()
        
        try:
            # Prepare request body
            request_body = {
                'notebook_file': {
                    'drive_file': {
                        'id': file_id
                    }
                },
                'output_notebook_file': {
                    'drive_file': {
                        'id': file_id  # Output to the same file
                    }
                },
                'parameters': parameters or {},
                'execution_options': {
                    'machine_type': machine_type,
                    'max_execution_duration_sec': max_execution_duration
                }
            }
            
            # Add accelerator configuration if requested
            if accelerator_type and accelerator_count > 0:
                request_body['execution_options']['accelerator_config'] = {
                    'type': accelerator_type,
                    'core_count': accelerator_count
                }
            
            # Execute the notebook
            execution = self.colab_service.notebooks().execute(
                body=request_body,
                timeout=timeout
            ).execute()
            
            execution_id = execution.get('name')
            logger.info(f"Started notebook execution with ID: {execution_id}")
            return execution_id
            
        except HttpError as e:
            if e.resp.status == 429:
                message = "Resource quota exceeded. Please try again later."
                logger.error(message)
                raise ResourceExceededError(message) from e
            else:
                message = f"Failed to execute notebook: {str(e)}"
                logger.error(message)
                raise ExecutionError(message) from e
        except Exception as e:
            message = f"Failed to execute notebook: {str(e)}"
            logger.error(message)
            raise ExecutionError(message) from e
    
    def get_execution_status(self, execution_id: str) -> Dict[str, Any]:
        """
        Get the status of a notebook execution.
        
        Args:
            execution_id: The execution ID from execute_notebook.
            
        Returns:
            Dict: Status information about the execution.
            
        Raises:
            ExecutionError: If getting status fails.
        """
        if not self.colab_service:
            self._authenticate()
        
        try:
            response = self.colab_service.executions().get(
                name=execution_id
            ).execute()
            
            logger.debug(f"Execution status: {response.get('state')}")
            return response
            
        except Exception as e:
            message = f"Failed to get execution status: {str(e)}"
            logger.error(message)
            raise ExecutionError(message) from e
    
    def wait_for_execution(
        self, 
        execution_id: str, 
        poll_interval: int = POLL_INTERVAL,
        max_wait_time: int = MAX_EXECUTION_WAIT_TIME
    ) -> Dict[str, Any]:
        """
        Wait for notebook execution to complete.
        
        Args:
            execution_id: The execution ID from execute_notebook.
            poll_interval: How often to check status in seconds.
            max_wait_time: Maximum time to wait in seconds.
            
        Returns:
            Dict: Final status information about the execution.
            
        Raises:
            ExecutionError: If execution fails or times out.
        """
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            status = self.get_execution_status(execution_id)
            state = status.get('state')
            
            if state in ('SUCCEEDED', 'FAILED', 'CANCELLED'):
                if state == 'FAILED':
                    error = status.get('error', {})
                    message = f"Execution failed: {error.get('message', 'Unknown error')}"
                    logger.error(message)
                    raise ExecutionError(message)
                elif state == 'CANCELLED':
                    message = "Execution was cancelled"
                    logger.warning(message)
                    raise ExecutionError(message)
                
                logger.info(f"Execution completed with state: {state}")
                return status
            
            logger.debug(f"Execution in progress, state: {state}")
            time.sleep(poll_interval)
        
        message = f"Execution timed out after {max_wait_time} seconds"
        logger.error(message)
        raise ExecutionError(message)
    
    def download_execution_result(
        self, 
        execution_id: str, 
        output_path: Optional[str] = None
    ) -> str:
        """
        Download the result of a notebook execution.
        
        Args:
            execution_id: The execution ID from execute_notebook.
            output_path: Path where to save the notebook. If None, a temporary
                        file will be created.
            
        Returns:
            str: Path to the downloaded notebook file.
            
        Raises:
            ExecutionError: If download fails.
        """
        if not self.colab_service or not self.drive_service:
            self._authenticate()
        
        try:
            # Get the execution details to find the output file
            execution = self.get_execution_status(execution_id)
            
            if execution.get('state') != 'SUCCEEDED':
                message = f"Cannot download results: execution not successful (state: {execution.get('state')})"
                logger.error(message)
                raise ExecutionError(message)
            
            # Get the output file ID
            output_file = execution.get('output_notebook_file', {}).get('drive_file', {}).get('id')
            
            if not output_file:
                message = "No output file found in execution result"
                logger.error(message)
                raise ExecutionError(message)
            
            # Create output path if not provided
            if not output_path:
                output_dir = tempfile.gettempdir()
                output_path = os.path.join(output_dir, f"colab_result_{int(time.time())}.ipynb")
            
            # Download the file
            request = self.drive_service.files().get_media(fileId=output_file)
            
            with open(output_path, 'wb') as f:
                downloader = MediaIoBaseDownload(f, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                    logger.debug(f"Download progress: {int(status.progress() * 100)}%")
            
            logger.info(f"Downloaded execution result to {output_path}")
            return output_path
            
        except Exception as e:
            message = f"Failed to download execution result: {str(e)}"
            logger.error(message)
            raise ExecutionError(message) from e
    
    def list_notebooks(
        self, 
        folder_id: Optional[str] = None,
        query: Optional[str] = None,
        max_results: int = 100
    ) -> List[Dict[str, Any]]:
        """
        List notebooks in Google Drive, optionally filtered by folder or query.
        
        Args:
            folder_id: ID of folder to list notebooks from.
            query: Optional search query to filter results.
            max_results: Maximum number of results to return.
            
        Returns:
            List[Dict]: List of notebook information dictionaries.
        """
        if not self.drive_service:
            self._authenticate()
        
        try:
            # Build the query
            q_parts = ["mimeType='application/x-ipynb+json'"]
            
            if folder_id:
                q_parts.append(f"'{folder_id}' in parents")
                
            if query:
                q_parts.append(f"(name contains '{query}' or fullText contains '{query}')")
            
            q = " and ".join(q_parts)
            
            # Execute the request
            response = self.drive_service.files().list(
                q=q,
                spaces='drive',
                fields='files(id, name, createdTime, modifiedTime, description, webViewLink)',
                pageSize=max_results
            ).execute()
            
            notebooks = response.get('files', [])
            logger.info(f"Found {len(notebooks)} notebooks matching criteria")
            return notebooks
            
        except Exception as e:
            logger.error(f"Failed to list notebooks: {str(e)}")
            return []
    
    def delete_notebook(self, file_id: str) -> bool:
        """
        Delete a notebook from Google Drive.
        
        Args:
            file_id: ID of the notebook to delete.
            
        Returns:
            bool: True if deletion was successful, False otherwise.
        """
        if not self.drive_service:
            self._authenticate()
        
        try:
            self.drive_service.files().delete(fileId=file_id).execute()
            logger.info(f"Deleted notebook with ID: {file_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete notebook: {str(e)}")
            return False
    
    def share_notebook(
        self, 
        file_id: str, 
        email: str, 
        role: str = 'reader',
        send_notification: bool = True,
        message: Optional[str] = None
    ) -> bool:
        """
        Share a notebook with a specific user.
        
        Args:
            file_id: ID of the notebook to share.
            email: Email address of the user to share with.
            role: Permission role ('reader', 'commenter', 'writer', 'owner').
            send_notification: Whether to send notification email.
            message: Optional message to include in notification.
            
        Returns:
            bool: True if sharing was successful, False otherwise.
        """
        if not self.drive_service:
            self._authenticate()
        
        try:
            permission = {
                'type': 'user',
                'role': role,
                'emailAddress': email
            }
            
            self.drive_service.permissions().create(
                fileId=file_id,
                body=permission,
                sendNotificationEmail=send_notification,
                emailMessage=message
            ).execute()
            
            logger.info(f"Shared notebook {file_id} with {email} as {role}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to share notebook: {str(e)}")
            return False
    
    def get_execution_metrics(self, execution_id: str) -> Dict[str, Any]:
        """
        Get performance metrics for a notebook execution.
        
        Args:
            execution_id: The execution ID from execute_notebook.
            
        Returns:
            Dict: Metrics information such as runtime, memory usage, etc.
            
        Raises:
            ExecutionError: If getting metrics fails.
        """
        if not self.colab_service:
            self._authenticate()
        
        try:
            status = self.get_execution_status(execution_id)
            
            # Extract metrics from the status
            metrics = {
                'state': status.get('state'),
                'creation_time': status.get('createTime'),
                'start_time': status.get('startTime'),
                'end_time': status.get('endTime'),
            }
            
            # Calculate runtime if available
            if metrics['start_time'] and metrics['end_time']:
                start = self._parse_timestamp(metrics['start_time'])
                end = self._parse_timestamp(metrics['end_time'])
                if start and end:
                    metrics['runtime_seconds'] = (end - start).total_seconds()
            
            # Add resource usage info if available
            if 'execution_stats' in status:
                metrics['execution_stats'] = status['execution_stats']
            
            return metrics
            
        except Exception as e:
            message = f"Failed to get execution metrics: {str(e)}"
            logger.error(message)
            raise ExecutionError(message) from e
    
    def _parse_timestamp(self, timestamp_str: Optional[str]) -> Optional[Any]:
        """
        Parse a timestamp string from Google API.
        
        Args:
            timestamp_str: Timestamp string in RFC 3339 format.
            
        Returns:
            datetime: Parsed timestamp or None if parsing fails.
        """
        if not timestamp_str:
            return None
            
        from datetime import datetime
        try:
            # Handle Z (UTC) suffix
            if timestamp_str.endswith('Z'):
                timestamp_str = timestamp_str[:-1] + '+00:00'
            
            return datetime.fromisoformat(timestamp_str)
        except ValueError:
            logger.warning(f"Could not parse timestamp: {timestamp_str}")
            return None
