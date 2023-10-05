import requests
import html

DEFAULT_CHAT_PARAMS = {
    'user_input': 'Say something interesting',
    'history': [],
    'context_instruct': '', # Optional

    'max_new_tokens': 500,
    'auto_max_new_tokens': False,
    'max_tokens_second': 0,
    'mode': 'instruct',  # Valid options: 'chat', 'chat-instruct', 'instruct'
    'character': 'Example',
    'instruction_template': 'Orca Mini',  # Will get autodetected if unset
    'your_name': 'You',
    # 'name1': 'name of user', # Optional
    # 'name2': 'name of character', # Optional
    # 'greeting': 'greeting', # Optional
    # 'name1_instruct': 'You', # Optional
    # 'name2_instruct': 'Assistant', # Optional
    # 'context_instruct': 'context_instruct', # Optional
    # 'turn_template': 'turn_template', # Optional
    'regenerate': False,
    '_continue': False,
    'chat_instruct_command': 'Continue the chat dialogue below. Write a single reply for the character "<|character|>".\n\n<|prompt|>',

    # Generation params. If 'preset' is set to different than 'None', the values
    # in presets/preset-name.yaml are used instead of the individual numbers.
    'preset': 'None',
    'do_sample': True,
    'temperature': 0.7,
    'top_p': 0.3,
    'typical_p': 1,
    'epsilon_cutoff': 0,  # In units of 1e-4
    'eta_cutoff': 0,  # In units of 1e-4
    'tfs': 1,
    'top_a': 0,
    'repetition_penalty': 1.18,
    'repetition_penalty_range': 0,
    'top_k': 40,
    'min_length': 0,
    'no_repeat_ngram_size': 0,
    'num_beams': 1,
    'penalty_alpha': 0,
    'length_penalty': 1,
    'early_stopping': False,
    'mirostat_mode': 0,
    'mirostat_tau': 5,
    'mirostat_eta': 0.1,
    'guidance_scale': 1,
    'negative_prompt': '',

    'seed': -1,
    'add_bos_token': True,
    'truncation_length': 2048,
    'ban_eos_token': False,
    'skip_special_tokens': True,
    'stopping_strings': []
}

def send_request(
    endpoint: str,
    messages: dict[str, str | dict[str, list[list[str]]]],
    temperature: float,
    max_tokens: int,
) -> str:
    params = {**DEFAULT_CHAT_PARAMS}
    params['user_input'] = messages['user_input']
    params['history'] = messages['history']
    params['context_instruct'] = messages['context_instruct']
    params['temperature'] = temperature
    params['max_new_tokens'] = max_tokens

    response = requests.post(endpoint, json=params)

    if response.status_code == 200:
        result = response.json()['results'][0]['history']
        output = html.unescape(result['visible'][-1][1])
        return output
    else:
        raise Exception(f'Response returned with status [{response.status_code}]')
    
