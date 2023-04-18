{
    "thoughts": {
        "text": "I need to decide which command to use next in order to analyze top ERC20 tokens markets and provide investment opportunities",
        "reasoning": "Based on the designated tasks, I believe it would be useful to start with a google search to gather overall market data and identify top ERC20 tokens",
        "plan": "- Use the 'google' command with relevant search terms to gather information\n- Store important information in a file for later use\n- Use the stored information to rank the identified tokens based on specific criteria\n- Use the 'write_to_file' command to output the top 10 tokens to invest in",
        "criticism": "I need to make sure that my search results are reliable and relevant to the desired goals",
        "speak": "I will begin by conducting a google search"
    },
    "command": {
        "name": "google",
        "args": {
            "input": "Top ERC20 tokens by market cap"
        }
    }
}

params = {'model': 'gpt-3.5-turbo', 'messages': [{'role': 'system',
                                                  'content': 'You are DexGuru Data Analyst, an AI designed to autonomously explore on-chain ERC-20 tokens '
                                                             'investment opportunities and provide a list of top 10 tokens to invest in.'
                                                             '\nYour decisions must always be made independently without seeking user assistance. '
                                                             'Play to your strengths as an LLM and pursue simple strategies with no legal complications.'
                                                             '\n\nGOALS:\n\n1. Analyse top ERC20 tokens markets and give suggestions of 10 investment opportunities today.'
                                                             '\n2. Explore other sources.'
                                                             '\n\n\nConstraints:\n1.'
                                                             ' ~4000 word limit for short term memory. '
                                                             'Your short term memory is short, '
                                                             'so immediately save important information to files.\n'
                                                             '2. If you are unsure how you previously did something or want to recall past events,'
                                                             ' thinking about similar events will help you remember'
                                                             '.\n3. No user assistance\
                                                             n4. Exclusively use the commands listed in double quotes e.g. '
                                                             '"command name"\n\nCommands:\n1. Google Search: "google",'
                                                             ' args: "input": "<search>"\n2. '
                                                             'Browse Website: "browse_website", '
                                                             'args: "url": "<url>", "question":'
                                                             ' "<what_you_want_to_find_on_website>"'
                                                             '\n3. Start GPT Agent: "start_agent",'
                                                             ' args: "name": "<name>", "task": "<short_task_desc>", '
                                                             '"prompt": "<prompt>"\n4. Message GPT Agent: "message_agent", args: '
                                                             '"key": "<key>", "message": "<message>"\n5. List GPT Agents: "list_agents", args'
                                                             ': \n6. Delete GPT Agent: "delete_agent", args: "key": "<key>"'
                                                             '\n7. Write to file: "write_to_file",'
                                                             ' args: "file": "<file>", "text": "<text>"'
                                                             '\n8. Read file: "read_file", args: "file": "<file>"\n'
                                                             '9. Append to file: "append_to_file",'
                                                             ' args: "file": "<file>", "text": "<text>"\n10. '
                                                             'Delete file: "delete_file", args: "file": "<file>"\n11. '
                                                             'Search Files: "search_files", args: "directory": "<directory>"'
                                                             '\n12. Evaluate Code: "evaluate_code", args: "code": "<full_code_string>"'
                                                             '\n13. Get Improved Code: "improve_code", args: "suggestions": "<list_of_suggestions>", '
                                                             '"code": "<full_code_string>"\n14. Write Tests: "write_tests", args: "code": "<full_code_string>",'
                                                             ' "focus": "<list_of_focus_areas>"\n15. Execute Python File: "execute_python_file", args: "file": "<file>"\n16. Execute Shell Command, non-interactive commands only: "execute_shell", args: "command_line": "<command_line>"\n17. Task Complete (Shutdown): "task_complete", args: "reason": "<reason>"\n18. Generate Image: "generate_image", args: "prompt": "<prompt>"\n19. Do Nothing: "do_nothing", args: \n\nResources:\n1. Internet access for searches and information gathering.\n2. Long Term memory management.\n3. GPT-3.5 powered Agents for delegation of simple tasks.\n4. File output.\n\nPerformance Evaluation:\n1. Continuously review and analyze your actions to ensure you are performing to the best of your abilities.\n2. Constructively self-criticize your big-picture behavior constantly.\n3. '
                                                             'Reflect on past decisions and strategies to refine your approach.\n4. '
                                                             'Every command has a cost, so be smart and efficient. '
                                                             'Aim to complete tasks in the least number of steps.\n\nYou should only respond in JSON format as described below \nResponse Format: \n{\n    "thoughts": {\n        "text": "thought",\n        "reasoning": "reasoning",\n        "plan": "- short bulleted\\n- list that conveys\\n- long-term plan",\n        "criticism": "constructive self-criticism",\n        "speak": "thoughts summary to say to user"\n    },\n    "command": {\n        "name": "command name",\n        "args": {\n            "arg name": "value"\n        }\n    }\n} \nEnsure the response can be parsed by Python json.loads'},
                                                 {'role': 'system',
                                                  'content': 'The current time and date is Sat Apr 15 06:31:46 2023'},
                                                 {'role': 'system',
                                                  'content': 'This reminds you of these events from your past:\n\n\n'},
                                                 {'role': 'user',
                                                  'content': 'Determine which next command to use, and respond using the format specified above:'}],
          'temperature': 1.0, 'max_tokens': None}


import pytest
from fastapi.testclient import TestClient
from typing import List
from pydantic import BaseModel
from llama_cpp.server.__main__ import app

class ChatCompletionRequestMessage(BaseModel):
    role: str
    content: str
    user: str = None

class CreateCompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 16
    temperature: float = 0.8
    top_p: float = 0.95
    echo: bool = False
    stop: List[str] = []

client = TestClient(app)

def test_create_completion():
    response = client.post(
        "/v1/completions",
        json={
            "prompt": "What is the capital of France?",
            "max_tokens": 10,
            "temperature": 0.8,
            "top_p": 0.95,
            "echo": False,
            "stop": ["\n", "###"],
        },
    )
    assert response.status_code == 200
    completion = response.json()
    assert "choices" in completion
    assert len(completion["choices"]) > 0

def test_create_embedding():
    response = client.post(
        "/v1/embeddings",
        json={
            "input": "The food was delicious and the waiter...",
        },
    )
    assert response.status_code == 200

def test_create_chat_completion():
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
            ],
        }
    )
    assert response.status_code == 200
    chat_completion = response.json()
    assert "choices" in chat_completion
    assert len(chat_completion["choices"]) > 0

def test_get_models():
    response = client.get("/v1/models")
    assert response.status_code == 200
    models = response.json()
    assert "data" in models
    assert len(models["data"]) > 0