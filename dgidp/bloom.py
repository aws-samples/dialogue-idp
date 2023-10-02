import os
# BLOOM

hf_api_token = os.environ.get('HF_API_TOKEN')
API_URL = "https://api-inference.huggingface.co/models/bigscience/bloom"
headers = {"Authorization": "Bearer " + hf_api_token}

def query_hf_api(payload):
  response = requests.request("POST", API_URL, json=payload, headers={"Authorization": f"Bearer {hf_api_token}"})
  return response.json()
  #return json.load(response.content.decode("utf-8"))

def truncate_string(input_string, stop_token):
    tokens = input_string.split() # Split the input string into tokens
    output_tokens = [] # Create an empty list to store the output tokens

    # Loop through the tokens and add them to the output list until the stop token is encountered
    for token in tokens:
        if token == stop_token:
            break
        else:
            output_tokens.append(token)

    output_string = " ".join(output_tokens) # Join the output tokens into a string

    return output_string

def bloom_inference(input_sentence, max_length, sample_or_greedy, seed=42):
    if sample_or_greedy == "Sample":
        json_ = {
          "inputs": ''.join(input_sentence),
          "parameters": {
            "max_new_tokens": max_length,
            "top_p": 0.9,
            "do_sample": True,
            "seed": seed,
            "early_stopping": False,
            "length_penalty": 0.0,
            "eos_token_id": None,
          },
          "options":
          {
            "use_cache": False,
            "wait_for_model":True
          },
        }
    else:
        json_ = {
          "inputs": ''.join(input_sentence),
          "parameters": {
            "max_new_tokens": max_length,
            "do_sample": False,
            "seed": seed,
            "early_stopping": False,
            "length_penalty": 0.0,
            "eos_token_id": None,
          },
          "options":
          {
            "use_cache": False,
            "wait_for_model":True
          },
        }

    #payload = {"inputs": input_sentence, "parameters": parameters,"options" : {"use_cache": False} }
    data = query_hf_api(json_)

    if "error" in data:
        return (None, None, f"<span style='color:red'>ERROR: {data['error']} </span>")

    generation = data[0]["generated_text"].split(input_sentence, 1)[1]
    str_1 = generation.replace("\n", "").replace('\"','')
    return truncate_string(str_1, '^[\\s\\|]+$')
    '''
    return (
        before_prompt
        + input_sentence
        + prompt_to_generation
        + generation
        + after_generation,
        data[0]["generated_text"],
        "",
    )
    '''