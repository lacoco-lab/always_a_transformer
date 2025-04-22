import asyncio
import requests

from time import sleep
from tqdm.asyncio import tqdm

from more_itertools import chunked

from typing import Any, Dict, List, Iterable, Tuple, AsyncGenerator
from utils.inference_constants import INSTRUCT_INFERENCE_PARAMS, COMPLETION_INFERENCE_PARAMS


async def get_first_model_id(client: Any) -> str:
    async for model in client.models.list():
        return model.id
    raise RuntimeError("No models found")


def wait_for_engine_to_start(server_url: str, secs: int = 10):
    """ Wait for the vLLM server to be available.
    Attempt to check the health end point, if it returns 200. print the same

    server_url: The base URL of the vLLM server (typically ending with 'v1')
    secs: The number of seconds to wait between retries (default is 5 seconds)

    """
    health_server_url = server_url.replace('v1', 'health')
    while True:
        try:
            response = requests.get(f"{health_server_url}", verify=False)
            if response.status_code == 200:
                print(f"\n\nvLLM server is available now!\n\n")
                break
        except requests.exceptions.ConnectionError:
            print(f"\n\nWaiting for vLLM server to be available, retrying in {secs} seconds\n\n")
            sleep(secs)


async def openai_vllm_chat(
    client: Any, 
    model_id: Any,
    task_prompt: str, 
    system_prompt: str, 
    inference_params: Dict[str, Any], 
    xid: str = "task"
) -> Any:
    """Send a single chat request to the vLLM server and await the response.
    
    This function fetches available models and then creates a chat completion
    using the provided prompts. It handles the asynchronous API calls and 
    returns the complete response.
    
    Args:
        client: The OpenAI-compatible client for making API calls
        task_prompt: The user message content to send
        system_prompt: The system message content to send
        inference_params: Parameters to control inference behavior
        xid: Unique identifier for this request. Defaults to "task".
        
    Returns:
        The complete response object from the chat completion API
    
    Raises:
        Exception: If the API call fails for any reason
    """
    try:
    
        # Prepare the messages for the chat completion
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_prompt},
        ]
        response = await client.chat.completions.create(
            model=model_id,
            messages=messages,
            **inference_params,
            extra_headers={
                "x-request-id": xid,
            }
        )
        return response

    except Exception as e:
        print(f"Error in batched call for request {str(xid)}: {str(e)}")
        raise


async def _prepare_prompts(
    data: Iterable[str],
    task_prompt: Any,
    spaced: bool=False
) -> AsyncGenerator[Tuple[int, str], None]:
    """ For all the data, convert the string into a list of characters & add a space between them. 
    Was maybe useful for some earlier kinds of prompting, if `spaced` is True. 

   Args:
        data: Iterable containing input strings to be formatted
        task_prompt: Template object with a text() method for formatting prompts
        spaced: If True, join the characters of each input string with spaces
               before formatting. Defaults to False.
               
    Yields:
        Tuples of (index, formatted_prompt) for each item in the data
    """
    for idx, d in enumerate(data):
        yield idx, task_prompt.text({"input": " ".join(list(d.strip())) if spaced else d.strip()})


async def openai_single_chat(
    data: Iterable[str], 
    client: Any,
    model_id: Any, 
    task_prompt: Any, 
    system_prompt: Any, 
    spaced_input: bool = False, 
    xid: str = "task-{}", 
    inference_params: Dict[str, Any] = INSTRUCT_INFERENCE_PARAMS
) -> List[Any]:
    """Process a batch of prompts as chat requests asynchronously.
    
    This function creates a task for each prompt in the data, sends them to the
    vLLM server as chat requests, waits for all to complete, and returns the
    collected responses.
    
    Args:
        data: Iterable containing input strings to be processed
        client: The OpenAI-compatible client for making API calls
        task_prompt: Template object for formatting user prompts
        system_prompt: Template object for system messages
        spaced_input: If True, join characters with spaces in the input data
        xid: Template string for request IDs, formatted with the data index
        inference_params: Parameters to control inference behavior
        
    Returns:
        List of response objects from the chat completion API, one per input
    """
    q_tasks = []
    async for d_idx, d in _prepare_prompts(data, task_prompt, spaced=spaced_input):
        q_task = asyncio.create_task(openai_vllm_chat(client, model_id, d, system_prompt.text(), inference_params,
                                                      xid=xid.format(d_idx)))
        q_tasks.append(q_task)

    await tqdm.gather(*q_tasks)
    responses = [q_task.result() for q_task in q_tasks]
    return responses


def batch_chat(
    data: Iterable[str], 
    client: Any,
    task_prompt: Any, 
    system_prompt: Any, 
    batch_size: int = 1000, 
    inference_params: Dict[str, Any] = INSTRUCT_INFERENCE_PARAMS
) -> List[Any]:
    """Process a large dataset in batches using chat completion.
    
    This function divides the input data into batches of specified size,
    processes each batch asynchronously using openai_single_chat, and
    collects all responses.
    
    Args:
        data: Iterable containing all input strings to be processed
        client: The OpenAI-compatible client for making API calls
        task_prompt: Template object for formatting user prompts
        system_prompt: Template object for system messages
        batch_size: Maximum number of items to process in each batch
        inference_params: Parameters to control inference behavior
        
    Returns:
        List of all response objects from the chat completion API
    """    
    responses = []
    # Create a list of lists where each sub list has batch_size number of data samples 
    chunked_data = list(chunked(data, batch_size))
    # First, get the available models (this is an async operation)
    model_id = asyncio.run(get_first_model_id(client))

    for chunk in tqdm(chunked_data):
        # Process the current chunk of data asynchronously using the openai_single_chat function
        resp = asyncio.run(openai_single_chat(chunk, client, task_prompt, system_prompt, inference_params=inference_params))
        # Add the responses from the current chunk to the overall responses list
    return responses


async def openai_vllm_complete(
    client: Any,
    model_id: Any, 
    task_prompt: str, 
    inference_params: Dict[str, Any], 
    xid: str = "task"
) -> Any:
    """Send a single chat request to the vLLM server for completion models
    This function fetches available models, creates a chat completion, handles the 
    asynchronous API calls and returns the complete response.
    
    Args:
        client: The OpenAI-compatible client for making API calls
        task_prompt: The user message content to send
        inference_params: Parameters to control inference behavior
        xid: Unique identifier for this request. Defaults to "task".
        
    Returns:
        The complete response object from the chat completion API
    """
    response = await client.completions.create(
        model=model_id,
        prompt=task_prompt,
        **inference_params,
        extra_headers={
            "x-request-id": xid,
        }
    )
    return response


async def openai_single_complete(
    data: Iterable[str], 
    client: Any,
    model_id: Any, 
    task_prompt: Any, 
    xid: str = "task-{}", 
    inference_params: Dict[str, Any] = COMPLETION_INFERENCE_PARAMS
) -> List[Any]:
    """Process a batch of prompts as completion requests asynchronously.
    
    This function creates a task for each prompt in the data, sends them to the
    vLLM server as completion requests, waits for all to complete, and returns 
    the collected responses.
    
    KEY DIFFERENCE from `openai_single_chat` is that this function has no system prompts
    
    Args:
        data: Iterable containing input strings to be processed
        client: The OpenAI-compatible client for making API calls
        task_prompt: Template object for formatting prompts
        xid: Template string for request IDs, formatted with the data index
        inference_params: Parameters to control completion behavior
        
    Returns:
        List of response objects from the completion API, one per input
    """    
    q_tasks = []
    async for d_idx, d in _prepare_prompts(data, task_prompt):
        q_task = asyncio.create_task(openai_vllm_complete(client, model_id, d, inference_params, xid=xid.format(d_idx)))
        q_tasks.append(q_task)

    await tqdm.gather(*q_tasks)
    responses = [q_task.result() for q_task in q_tasks]
    return responses


def batch_complete(
    data: Iterable[str], 
    client: Any, 
    task_prompt: Any, 
    batch_size: int = 1000, 
    inference_params: Dict[str, Any] = COMPLETION_INFERENCE_PARAMS
) -> List[Any]:
    """Process a large dataset in batches using completion models.
    
    KEY DIFFERENCE from `batch_chat` is that this function has no system prompts
    
    This function divides the input data into batches of specified size,
    processes each batch asynchronously using openai_single_complete, and
    collects all responses.
    
    Args:
        data: Iterable containing all input strings to be processed
        client: The OpenAI-compatible client for making API calls
        task_prompt: Template object for formatting prompts
        batch_size: Maximum number of items to process in each batch
        inference_params: Parameters to control completion behavior
        
    Returns:
        List of all response objects from the completion API
    """    
    # Similar to batch chat.
    responses = []
    model_id = asyncio.run(get_first_model_id(client))
    chunked_data = list(chunked(data, batch_size))
    for chunk in tqdm(chunked_data):
        # Create a new event loop for each batch and process all items in parallel
        resp = asyncio.run(openai_single_complete(chunk, client, model_id, task_prompt, inference_params=inference_params))
        responses.extend(resp)
    return responses
