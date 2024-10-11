import asyncio
import os

from weave.flow.scorer import Scorer

import orchestrator as orc
import weave
from weave import Dataset
from openai import ChatCompletion
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

@weave.op()
def gen_weave_dataset_from_traces(client):
    pos_feedback = client.feedback(reaction="👍")
    neg_feedback = client.feedback(reaction="👎")

    pos_calls = pos_feedback.refs().calls()
    if len(list(pos_calls)) > 0:
        i = 0
        data = []
        for call in pos_calls:
            cc: ChatCompletion = call.output
            feed = call.feedback[0].payload['emoji']
            data.append({'id': i, 'messages': orc.build_message("user",call.inputs['messages']['content']),
                         'output': cc.choices[0].message.content,
                         'feedback': feed})
            i += 1
        pos_dataset = Dataset(name='Positive Feedback Interactions', rows=data)
        weave.publish(pos_dataset)
    else:
        pos_dataset = None

    neg_calls = neg_feedback.refs().calls()
    if len(list(neg_calls)) > 0:
        i = 0
        data = []
        for call in neg_calls:
            cc: ChatCompletion = call.output
            feed = call.feedback[0].payload['emoji']
            data.append({'id': i, 'messages': orc.build_message("user",call.inputs['messages']['content']),
                         'output': cc.choices[0].message.content,
                         'feedback': feed})
            i += 1
        neg_dataset = Dataset(name='Negative Feedback Interactions', rows=data)
        weave.publish(neg_dataset)
    else:
        neg_dataset = None



    return pos_dataset, neg_dataset


@weave.op()
def evaluate_and_score(model_base, pos_dataset, neg_dataset):

    Addressing = success_in_addressing_scorer()
    Clarity = success_in_clarity_scorer()
    Consiseness = success_in_conciseness_scorer()

    print("Evaluating:...")
    if pos_dataset is not None:
        pos_evaluation = weave.Evaluation(
            name="Initial Evaluation of LLM Performance with Good User Reviews",
            dataset=pos_dataset,
            scorers=[
                Addressing,
                Clarity,
                Consiseness

            ]
        )
        asyncio.run(pos_evaluation.evaluate(model_base))
    if neg_dataset is not None:
        neg_evaluation = weave.Evaluation(
            name="Initial Evaluation of LLM Performance with Bad User Reviews",
            dataset=neg_dataset,
            scorers=[
                Addressing,
                Clarity,
                Consiseness
            ]
        )
        asyncio.run(neg_evaluation.evaluate(model_base))


class success_in_addressing_scorer(Scorer):

    @weave.op()
    async def score(self, messages, model_output):
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
            "OpenAI_GPT_4o_Score": score,
        }


class success_in_clarity_scorer(Scorer):

    @weave.op()
    async def score(self, messages, model_output):
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
            "OpenAI_GPT_4o_Score": score,
        }


class success_in_conciseness_scorer(Scorer):

    @weave.op()
    async def score(self, messages, model_output):
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
            "OpenAI_GPT_4o_Score": score,
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