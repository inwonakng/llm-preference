from __future__ import annotations
import yaml
import tiktoken
encoding = tiktoken.get_encoding("cl100k_base")

from chat_api import DEFAULT_CHAT_PARAMS, send_request
import copy
DEFAULT_CHAT_PARAMS_summary = copy.deepcopy(DEFAULT_CHAT_PARAMS)
DEFAULT_CHAT_PARAMS_summary['max_new_tokens'] = 800
DEFAULT_CHAT_PARAMS_summary['temperature'] = 1
DEFAULT_CHAT_PARAMS_summary['top_p'] = 0.7

DELIMITERS = ['`', '"', '\'', '#', '.', '%']
def remove_delimiters(text: str) -> str:
    for delim in DELIMITERS:
        text = text.replace(delim, '')
    return text.strip()


class Task:
    text: str
    label: str
    alternative_a: str
    alternative_b: str

    def __init__(
        self,
        text: str,
        label: str,
        alternative_a: str,
        alternative_b: str,
    ) -> None:
        self.text = text
        self.label = label
        self.alternative_a = alternative_a
        self.alternative_b = alternative_b

class Prompt:
    instruction: str
    retry_msg: str
    examples: list
    task_template: str
    label_to_text: dict[int, str]
    text_to_label: dict[str, int]
    confirmation: list[str] | None = None
    examples: list[list[str]] = []

    def __init__(
        self,
        instruction: str,
        retry_msg: str,
        task_template: str,
        label_to_text: dict[int, str],
    ) -> None:
        self.instruction = instruction
        self.retry_msg = retry_msg
        self.task_template = task_template
        self.label_to_text = label_to_text
        self.text_to_label = {
            text.lower(): label 
            for label,text in label_to_text.items()
        }

    @staticmethod
    def load_template(template_path: str) -> Prompt:
        config = yaml.safe_load(open(template_path))
        prompt = Prompt(
            instruction = config['instruction'],
            retry_msg = config['retry_msg'],
            task_template = config['task'],
            label_to_text = config['label'],
        )
        if 'confirmation' in config:
            prompt.add_confirmation(config['confirmation'])
        return prompt

    def add_confirmation(self, confirmation: list[str]):
        self.confirmation = confirmation

    def wrap_task(self, task: Task) -> str:
        return self.task_template.replace(
            '{alternative_a}', task.alternative_a
        ).replace(
            '{alternative_b}', task.alternative_b
        ).replace(
            '{text}', task.text
        )
        
    def add_examples(
        self,
        examples: list[Task]
    ) -> None:
        self.examples = [
            [
                self.wrap_task(e),
                self.label_to_text[e.label]
            ]
            for e in examples
        ]

    def build(self, task: Task) -> dict[str, any]:
        if self.confirmation is not None:
            history = [self.confirmation]
        else:
            history = []
        if self.examples:
            history += self.examples
        return dict(
            **DEFAULT_CHAT_PARAMS,
            user_input = self.wrap_task(task),
            history = dict(
                internal = history,
                visible = history
            ),
            context_instruct = self.instruction,
        )
    
    def build_retry(self, task: Task, prev_response: str) -> dict[str, any]:
        if self.confirmation is not None:
            history = [self.confirmation]
        else:
            history = []
        if self.examples:
            history += self.examples
        history += [[self.wrap_task(task), prev_response]]
        return dict(
            **DEFAULT_CHAT_PARAMS,
            user_input = self.retry_msg,
            history = dict(
                internal = history,
                visible = history
            ),
            context_instruct = self.instruction,
        )

    def execute(
        self,
        task: Task,
        api_endpoint: str = 'http://localhost:5000/api/v1/chat',
        verbose: bool = False,
    ) -> int:
        params = self.build(task)
        output = send_request(api_endpoint, params)
        # if verbose:
        #     print('Comment:', task.text)
        #     print('A:', task.alternative_a)
        #     print('B:', task.alternative_b)
        #     print('True label:', self.label_to_text[task.label])
        #     print('Model output:', output)

        retry_cnt = 0
        while output.lower() not in self.text_to_label:
            retry_cnt += 1
            params = self.build_retry(task, output)
            output = send_request(api_endpoint, params)
            # if verbose:
            #     print('Retry output:', output)
            if retry_cnt == 10:
                return False, None, params
        
        return True, self.text_to_label[output.lower()], params

    def __str__(self) -> str:
        return f'instruction:\n{self.instruction}\nretry_msg:\n{self.retry_msg}\ntask_template:\n{self.task_template}\nlabel_to_text:\n{self.label_to_text}\nconfirmation:\n{self.confirmation}\ntwo_stage_instruction:{self.instruction_binary}\nself.label_to_text_binary:{self.label_to_text_binary}\nexamples:\n{self.examples}'
    

class ThreeStagePrompt:
    instruction: str 
    question_1st_stage: str
    question_2nd_stage: str
    instruction_summary: str ### added by linj26 
    retry_msg_1st_stage: str
    retry_msg_2nd_stage: str
    retry_msg_summary: str ### added by linj26 
    examples: list
    task_template: str
    label_to_text_1st_stage: dict[int, str]
    text_to_label_1st_stage: dict[str, int]
    label_to_text_2nd_stage: dict[int, str]
    text_to_label_2nd_stage: dict[str, int]
    confirmation: list[str] | None = None
    examples: list[list[str]] = []
    example_summary: list ### added by linj26 
    synonyms: dict[str, list[str]] ### added by linj26 

    def __init__(
        self,
        instruction: str, ### added by linj26 
        question_1st_stage: str,
        question_2nd_stage: str  ,      
        instruction_summary: str, ### added by linj26 
        retry_msg_1st_stage: str,
        retry_msg_2nd_stage: str,
        retry_msg_summary: str, ### added by linj26 
        task_template: str,
        label_to_text: dict[int, str],
        label_binary: dict[int, str],
        example_summary: list, ### added by linj26 ###
        synonyms: dict[str, list[str]] ### added by linj26 

    ) -> None:
        self.instruction = instruction 
        self.question_1st_stage = question_1st_stage
        self.question_2nd_stage = question_2nd_stage
        self.instruction_summary = instruction_summary ### added by linj26 
        self.retry_msg_1st_stage = retry_msg_1st_stage
        self.retry_msg_2nd_stage = retry_msg_2nd_stage
        self.retry_msg_summary =  retry_msg_summary ### added by linj26 
        self.task_template = task_template
        self.label_to_text_1st_stage = label_binary
        self.text_to_label_1st_stage = {
            text.lower(): label 
            for label,text in label_binary.items()
        }
        self.label_to_text_2nd_stage = label_to_text
        self.text_to_label_2nd_stage = {
            text.lower(): label 
            for label,text in label_to_text.items()
        }
        self.example_summary = [] ### added by linj26 
        self.synonyms = {} ### added by linj26 

    @staticmethod
    def load_template(template_path: str) -> Prompt:
        config = yaml.safe_load(open(template_path))
        prompt = ThreeStagePrompt(
            instruction = config['instruction'], 
            question_1st_stage = config['question_1st_stage'], 
            question_2nd_stage = config['question_2nd_stage'], 
            instruction_summary = config['instruction_summary'], ### added by linj26 
            retry_msg_1st_stage = config['retry_msg_1st_stage'],
            retry_msg_2nd_stage = config['retry_msg_2nd_stage'],
            retry_msg_summary = config['retry_msg_summary'], ### added by linj26 
            task_template = config['task'],
            label_to_text = config['label'],
            label_binary = config['label_binary'],
            example_summary = [],
            synonyms = {},
        )
        if 'confirmation' in config:
            prompt.add_confirmation(config['confirmation'])
        return prompt
    
    def add_confirmation(self, confirmation: list[str]):
        self.confirmation = confirmation

    def wrap_task(self, task: Task) -> str:
        return self.task_template.replace(
            '{alternative_a}', task.alternative_a
        ).replace(
            '{alternative_b}', task.alternative_b
        ).replace(
            '{text}', task.text
        )
        
    def add_examples(
        self,
        examples: list[Task]
    ) -> None:
        self.examples = []
        for e in examples:
            self.examples.append([self.wrap_task(e) + self.question_1st_stage, self.label_to_text_1st_stage[e.label > 0]])
            self.examples.append([self.wrap_task(e) + self.question_2nd_stage, self.label_to_text_2nd_stage[e.label]])
    
    def build_two_stage_1(self, task: Task) -> dict[str, any]:
        if self.confirmation is not None:
            history = [self.confirmation]
        else:
            history = []
        if self.examples:
            history += self.examples
        params = dict(
            **DEFAULT_CHAT_PARAMS,
            user_input = self.wrap_task(task) + self.question_1st_stage,
            history = dict(
                internal = history,
                visible = history
            ),
            context_instruct = self.instruction,
        )

        input_1st = [params['user_input']] + [params['context_instruct']] 
        for i in range(len(history)):
            input_1st += [history[i][0]]
            input_1st += [history[i][0]]
            input_1st += [history[i][1]]
            input_1st += [history[i][1]]

        return params, input_1st
    
    def build_retry_two_stage_1(self, task: Task, prev_response: str) -> dict[str, any]:
        if self.confirmation is not None:
            history = [self.confirmation]
        else:
            history = []
        if self.examples:
            history += self.examples
        history += [[self.wrap_task(task) + self.question_1st_stage, prev_response]]
        params =  dict(
            **DEFAULT_CHAT_PARAMS,
            user_input = self.retry_msg_1st_stage,
            history = dict(
                internal = history,
                visible = history
            ),
            context_instruct = self.instruction,
        )

        input_retry_1st = [params['user_input']] + [params['context_instruct']] 
        for i in range(len(history)):
            input_retry_1st += [history[i][0]]
            input_retry_1st += [history[i][0]]
            input_retry_1st += [history[i][1]]
            input_retry_1st += [history[i][1]]
    
        return params, input_retry_1st
    
    ### added by linj26 ###
    def build_summary(self, task: Task) -> dict[str, any]:
        if self.confirmation is not None:
            history = [self.confirmation]
        else:
            history = []
        history += self.example_summary

        params = dict(
            **DEFAULT_CHAT_PARAMS_summary,
            user_input = self.wrap_task(task),
            history = dict(
                internal = history,
                visible = history
            ),
            context_instruct = self.instruction_summary,
        )

        input_summary = [params['user_input']] + [params['context_instruct']] 
        for i in range(len(history)):
            input_summary += [history[i][0]]
            input_summary += [history[i][0]]
            input_summary += [history[i][1]]
            input_summary += [history[i][1]]

        return params, input_summary
    
    
    ### added by linj26 ###
    def build_retry_summary(self, task: Task, prev_response: str) -> dict[str, any]:
        if self.confirmation is not None:
            history = [self.confirmation]
        else:
            history = []
        history += self.examples
        history += [[self.wrap_task(task), prev_response]]

        params = dict(
            **DEFAULT_CHAT_PARAMS_summary,
            user_input = self.retry_msg_summary,
            history = dict(
                internal = history,
                visible = history
            ),
            context_instruct = self.instruction_summary,
        )

        input_retry_summary = [params['user_input']] + [params['context_instruct']] 
        for i in range(len(history)):
            input_retry_summary += [history[i][0]]
            input_retry_summary += [history[i][0]]
            input_retry_summary += [history[i][1]]
            input_retry_summary += [history[i][1]]

        return params, input_retry_summary

    def build_two_stage_2(self, task: Task) -> dict[str, any]:
        if self.confirmation is not None:
            history = [self.confirmation]
        else:
            history = []
        if self.examples:
            history += self.examples
        history += [[self.wrap_task(task) + self.question_1st_stage, self.label_to_text_1st_stage[1]]]

        params = dict(
            **DEFAULT_CHAT_PARAMS,
            user_input = self.question_2nd_stage,
            history = dict(
                internal = history,
                visible = history
            ),
            context_instruct = self.instruction,
        )

        input_2nd = [params['user_input']] + [params['context_instruct']] 
        for i in range(len(history)):
            input_2nd += [history[i][0]]
            input_2nd += [history[i][0]]
            input_2nd += [history[i][1]]
            input_2nd += [history[i][1]]

        return params, input_2nd

    def build_retry_two_stage_2(self, task: Task, prev_response: str) -> dict[str, any]:
        if self.confirmation is not None:
            history = [self.confirmation]
        else:
            history = []
        if self.examples:
            history += self.examples
        history += [[self.wrap_task(task) + self.question_1st_stage, self.label_to_text_1st_stage[1]]]
        history += [[self.question_2nd_stage, prev_response]]

        params = dict(
            **DEFAULT_CHAT_PARAMS,
            user_input = self.retry_msg_2nd_stage,
            history = dict(
                internal = history,
                visible = history
            ),
            context_instruct = self.instruction,
        )

        input_retry_2nd = [params['user_input']] + [params['context_instruct']] 
        for i in range(len(history)):
            input_retry_2nd += [history[i][0]]
            input_retry_2nd += [history[i][0]]
            input_retry_2nd += [history[i][1]]
            input_retry_2nd += [history[i][1]]
    
        return params, input_retry_2nd
    
    ### added by linj26 ###
    def check_choices_exist(
        self, 
        alternative_a: list[str],
        alternative_b: list[str],
        output_summary: str
    ) -> bool:

        check_a = False
        check_b = False
        for i in alternative_a:
            if i in output_summary:
                check_a = True

        for i in alternative_b:
            if i in output_summary:
                check_b = True

        return not check_a or not check_b # both choices exist in summary
        # return not check_a and not check_b # one choices exist in summary

    ### added by linj26 ### 
    def update_example_summary(
        self, 
        ori_task_text: str,
        output: str,
        label: str,
        retry_cnt: int,
        alternative_a: str,
        alternative_b: str,
        output_summary: str,
    ):
        if (label) == (self.text_to_label_2nd_stage[output.lower()]) and retry_cnt == 0 :
            new_example = [f"Comment: {ori_task_text}\nChoices: {alternative_a} & {alternative_b} \nResponse: \n"] + [f'{output_summary}']
            self.example_summary += [new_example]
            print('update_example_summary')

        if len(self.example_summary) > 20:
            self.example_summary.pop(0)
        
    def checK_len_summary(
        self, 
        output_summary_temp: str,
        text: str,
    ):
        summary_too_long = False
        len_summary = len(encoding.encode(output_summary_temp))
        len_text = len(encoding.encode(text))
        if len_text < 50:
            if len_summary > 40: 
                summary_too_long = True
        elif 50 <= len_text < 100:
            if len_summary > 80: 
                summary_too_long = True
        elif 100 <= len_text :
            if len_summary > 150:
                summary_too_long = True
        return summary_too_long

    def execute(
        self,
        task: Task,
        length_todo_summary: int,
        api_endpoint: str = 'http://localhost:5000/api/v1/chat',
        verbose: bool = False,
    ) -> int:
        ### added by linj26 ###
        ''' ################## summarization ################## ''' 
        token_summary_input = []
        token_summary_output = []

        task_local = copy.deepcopy(task)
        task_text = task_local.text

        if len(encoding.encode(task_text)) > length_todo_summary:
            # print(f'11111111 task_local.text: {task_local.text}')
            params, input_summary = self.build_summary(task_local)
            # print(f'11111111 input_summary: {input_summary}')
            output_summary = send_request(api_endpoint, params)

            alternative_a = self.synonyms[task_local.alternative_a]
            alternative_b = self.synonyms[task_local.alternative_b]
            choices_in_summary = self.check_choices_exist(alternative_a, alternative_b, output_summary)

            # retry
            time_retry = 0
            output_summary_temp = output_summary
            input_retry_summary = []
            output_retry_summary = ''

            summary_too_long = self.checK_len_summary(output_summary_temp, task_text)

            while ( choices_in_summary or summary_too_long ) and time_retry < 5:
                time_retry += 1
                params, input_retry_summary_temp = self.build_retry_summary(task_local, output_summary_temp)
                input_retry_summary = input_retry_summary + input_retry_summary_temp
                output_summary_temp = send_request(api_endpoint, params)
                output_retry_summary = output_retry_summary + output_summary_temp
                choices_in_summary = self.check_choices_exist(alternative_a, alternative_b, output_summary_temp)
                if verbose:
                    print(f' Retry summary No.{time_retry}: [ output:{output_summary_temp} ]')
            if time_retry < 5: 
                output_summary = output_summary_temp # retry success
            if verbose:
                print(f'***Summary*** :{output_summary} ]')
                print('-'*20)

            token_summary_input = token_summary_input + input_summary + input_retry_summary
            token_summary_output.append(output_summary)
            token_summary_output.append(output_retry_summary)
            task_local.text = output_summary

        ''' ################## check preference ################## ''' 
        token_1st_input = []
        token_1st_output = []
        # print(f'2222222222 task_local.text: {task_local.text}')
        params, input_1st = self.build_two_stage_1(task_local)
        output = send_request(api_endpoint, params)

        retry_output = output
        retry_cnt = 0
        input_retry_1st = []
        output_retry_1st = ''
        while retry_output.lower() not in self.text_to_label_1st_stage:
            retry_cnt += 1
            params, input_retry_1st_temp = self.build_retry_two_stage_1(task_local, output)
            input_retry_1st = input_retry_1st + input_retry_1st_temp
            retry_output = send_request(api_endpoint, params)
            output_retry_1st = output_retry_1st + retry_output
            if verbose:
                print('stage1(check preference) retry output:', retry_output)
            if retry_cnt == 10:
                token_1st_input = token_1st_input + input_1st + input_retry_1st
                token_1st_output.append(output)
                token_1st_output.append(output_retry_1st)

                ### added by linj26 ### 
                '''update example_summary'''
                if len(encoding.encode(task_text)) > length_todo_summary:
                    self.update_example_summary(task_text, output, task_local.label,\
                        retry_cnt, task_local.alternative_a, task_local.alternative_b, output_summary)

                return False, None, params, token_summary_input, token_summary_output, token_1st_input, token_1st_output
        output = retry_output

        # if verbose:
        #     # print('Comment:', task.text)
        #     # print('A:', task.alternative_a)
        #     # print('B:', task.alternative_b)
        #     print('stage1 True label:', self.label_to_text_2nd_stage[task_local.label])
        #     print('stage1 model output:', output)

        token_1st_input = token_1st_input + input_1st + input_retry_1st
        token_1st_output.append(output)
        token_1st_output.append(output_retry_1st)

        if output.lower() == self.label_to_text_1st_stage[0].lower(): # No preference
            ### added by linj26 ### 
            '''update example_summary'''
            if len(encoding.encode(task_text)) > length_todo_summary:
                self.update_example_summary(task_text, output, task_local.label,\
                    retry_cnt, task_local.alternative_a, task_local.alternative_b, output_summary)

            return True, 0, params, token_summary_input, token_summary_output, token_1st_input, token_1st_output
        elif output.lower() == self.label_to_text_1st_stage[1].lower(): # has preference

            ''' ################## preference classification ################## ''' 
            token_2nd_input = []
            token_2nd_output = []
            # print(f'task_local.text: {task_local.text}')
            params, input_2nd = self.build_two_stage_2(task_local)
            output = send_request(api_endpoint, params)

            valid_outputs = {
                text.lower(): label 
                for label,text in self.label_to_text_2nd_stage.items() if label > 0
            }
            retry_output = output
            retry_cnt = 0
            input_retry_2nd = []
            output_retry_2nd = ''
            while retry_output.lower() not in valid_outputs:
                retry_cnt += 1
                params, input_retry_2nd_temp = self.build_retry_two_stage_2(task_local, output)
                input_retry_2nd = input_retry_2nd + input_retry_2nd_temp
                retry_output = send_request(api_endpoint, params)
                output_retry_2nd = output_retry_2nd + retry_output
                if verbose:
                    print('stage2(preference classification) retry output:', retry_output)
                if retry_cnt == 10:
                    token_2nd_input = token_2nd_input + input_2nd + input_retry_2nd
                    token_2nd_output.append(output)
                    token_2nd_output.append(output_retry_2nd)

                    ### added by linj26 ### 
                    '''update example_summary'''
                    if len(encoding.encode(task_text)) > length_todo_summary:
                        self.update_example_summary(task_text, output, task_local.label,\
                            retry_cnt, task_local.alternative_a, task_local.alternative_b, output_summary)

                    return False, None, params, token_summary_input, token_summary_output, token_1st_input + token_2nd_input,\
                        token_1st_output + token_2nd_output
            output = retry_output

            token_2nd_input = token_2nd_input + input_2nd + input_retry_2nd
            token_2nd_output.append(output)
            token_2nd_output.append(output_retry_2nd)

            if verbose:
                print('stage2 True label:', task_local.label)
                print('stage2 model output:', self.text_to_label_2nd_stage[output.lower()])

            ### added by linj26 ### 
            '''update example_summary'''
            if len(encoding.encode(task_text)) > length_todo_summary:
                self.update_example_summary(task_text, output, task_local.label,\
                    retry_cnt, task_local.alternative_a, task_local.alternative_b, output_summary)

            return True, self.text_to_label_2nd_stage[output.lower()], params, token_summary_input, token_summary_output,\
                token_1st_input + token_2nd_input, token_1st_output + token_2nd_output
        else:
            raise ValueError(f'Invalid 1st stage output {output}')

    def __str__(self) -> str:
        class_attributes = [attr for attr in dir(self) if not callable(getattr(self, attr))]
        return '\n'.join([f'{attr}:\n{getattr(self, attr)}' for attr in class_attributes])