import asyncio
import os

from weave.weave_client import WeaveClient

import orchestrator as orc
import weave
from weave import Dataset
from openai import ChatCompletion
from openai import OpenAI
import model_service as ms
from dotenv import load_dotenv
from datasets import load_dataset

load_dotenv()

@weave.op()
def gen_weave_dataset_from_traces(client: WeaveClient):
    pos_feedback = client.feedback(reaction="👍")
    neg_feedback = client.feedback(reaction="👎")

    pos_calls = pos_feedback.refs().calls()
    i = 0
    data = []
    for call in pos_calls:
        cc: ChatCompletion = weave.ref(call.output).get()
        feed = call.feedback[0].payload['emoji']
        data.append({'id': i, 'messages': orc.build_message("user",call.inputs['messages']['content']),
                     'output': cc.choices[0].message.content,
                     'feedback': feed})
        i += 1
    pos_dataset = Dataset(name='Positive Feedback Interactions', rows=data)

    neg_calls = neg_feedback.refs().calls()
    i = 0
    data = []
    for call in neg_calls:
        cc: ChatCompletion = weave.ref(call.output).get()
        feed = call.feedback[0].payload['emoji']
        data.append({'id': i, 'messages': orc.build_message("user",call.inputs['messages']['content']),
                     'output': cc.choices[0].message.content,
                     'feedback': feed})
        i += 1
    neg_dataset = Dataset(name='Negative Feedback Interactions', rows=data)

    weave.publish(pos_dataset)
    weave.publish(neg_dataset)

    return pos_dataset, neg_dataset


@weave.op()
def evaluate_and_score(base_prompt: str, client: WeaveClient):

    pos_dataset, neg_dataset = gen_weave_dataset_from_traces(client)

    model_base = ms.Llama_31_8B_Model(context=base_prompt)

    print("Evaluating:...")
    pos_evaluation = weave.Evaluation(
        name="Initial Evaluation of LLM Performance with Good User Reviews",
        dataset=pos_dataset,
        scorers=[
            success_in_adressing_scorer,
            success_in_clarity_scorer,
            success_in_conciseness_scorer

        ]
    )
    neg_evaluation = weave.Evaluation(
        name="Initial Evaluation of LLM Performance with Bad User Reviews",
        dataset=neg_dataset,
        scorers=[
            success_in_adressing_scorer,
            success_in_clarity_scorer,
            success_in_conciseness_scorer
        ]
    )
    eval_result_pos = asyncio.run(pos_evaluation.evaluate(model_base))
    eval_result_neg = asyncio.run(neg_evaluation.evaluate(model_base))

    return eval_result_pos, eval_result_neg


@weave.op()
def success_in_adressing_scorer(messages, model_output):
    context_evaluator_prompt = f'''
    You are a grader of LLM assistant responses.
    You will be given a prompt and an answer, and you must grade how well the answer
    addresses the prompt. You may only answer with float numbers between 0 and 1.
    No other output is allowed.

    prompt: {messages}
    answer: {model_output.choices[0].message.content}

'''

    messages = [
        orc.build_message("system", context_evaluator_prompt),
    ]

    response = call_oai(messages)
    print(response.choices[0].message.content)
    score = float(response.choices[0].message.content)
    return {
        "prompt_addressed_score": score,
    }

@weave.op()
def success_in_clarity_scorer(messages, model_output):
    context_evaluator_prompt = f'''
    You are a grader of LLM assistant responses.
    You will be given a prompt and an answer, and you must grade how clear the answer is when addressing the prompt.
    You may only answer with float numbers between 0 and 1.
    No other output is allowed.

    prompt: {messages}
    answer: {model_output.choices[0].message.content}

'''

    messages = [
        orc.build_message("system", context_evaluator_prompt),
    ]

    response = call_oai(messages)
    print(response.choices[0].message.content)
    score = float(response.choices[0].message.content)
    return {
        "prompt_addressed_score": score,
    }

@weave.op()
def success_in_conciseness_scorer(messages, model_output):
    context_evaluator_prompt = f'''
    You are a grader of LLM assistant responses.
    You will be given a prompt and an answer, and you must grade how the conciseness of the answer while addressing the prompt. 
    You may only answer with float numbers between 0 and 1.
    No other output is allowed.

    prompt: {messages}
    answer: {model_output.choices[0].message.content}

'''

    messages = [
        orc.build_message("system", context_evaluator_prompt),
    ]

    response = call_oai(messages)
    print(response.choices[0].message.content)
    score = float(response.choices[0].message.content)
    return {
        "prompt_addressed_score": score,
    }


@weave.op()
def call_oai(messages):
    oai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    try:
        completion = oai_client.chat.completions.create(
            model='gpt-4o',
            messages=messages,
            temperature=0.0,
        )
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e

    return completion