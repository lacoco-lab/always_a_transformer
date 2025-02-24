import openai

if __name__ == "__main__":
    base_url = f"http://0.0.0.0:8087/v1"
    client = openai.Client(base_url=base_url, api_key="sk_noreq", max_retries=10)
    model = client.models.list()
    print(model)

    messages = [
        {"role": "system", "content": "You are who you are."},
        {"role": "user", "content": "Who are you?"},
    ]
    inferece_params = {
        "seed": 5,
        "max_tokens": 3000,
        "temperature": 0.0,
        "logprobs": True,
    }
    response = client.chat.completions.create(
        model=model.data[0].id,
        # model=model,
        messages=messages,
        **inferece_params,
    )
    
    print(f"-----\n{response}\n-----")
    print(response.choices[0].message.content)
    print(f"Input length: {response.usage.prompt_tokens}")
    print(f"Output length: {response.usage.completion_tokens}")
    print(f"Tokenized Output: {[lp.token for lp in response.choices[0].logprobs.content]}")