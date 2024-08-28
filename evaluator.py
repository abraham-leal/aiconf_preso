import asyncio

import weave
from weave import Dataset
from openai import ChatCompletion
import model_service as ms

client = weave.init("abe_aiconf_llama31_8B_prompteng")
@weave.op()
def gen_weave_dataset_from_traces():
    pos_feedback = client.feedback(reaction="👍")
    neg_feedback = client.feedback(reaction="👎")

    pos_calls = pos_feedback.refs().calls()
    i = 0
    data = []
    for call, feedback in zip(pos_calls, pos_feedback):
        cc: ChatCompletion = weave.ref(call.output).get()
        data.append({'id': i, 'prompt': call.inputs['messages']['content'],'output': cc.choices[0].message.content, 'feedback': feedback.payload['emoji']})
        i += 1
    pos_dataset = Dataset(name='Positive Feedback Interactions', rows=data)

    neg_calls = neg_feedback.refs().calls()
    i = 0
    data = []
    for call, feedback in zip(neg_calls, neg_feedback):
        cc: ChatCompletion = weave.ref(call.output).get()
        data.append({'id': i, 'prompt': call.inputs['messages']['content'], 'output': cc.choices[0].message.content, 'feedback': feedback.payload['emoji']})
        i += 1
    neg_dataset = Dataset(name='Negative Feedback Interactions', rows=data)

    weave.publish(pos_dataset)
    weave.publish(neg_dataset)

# Define a scorer
@weave.op()
def success_in_timekeeping_scorer(messages, model_output):
    context_evaluator_prompt = f'''Given the prompt and answer verify if the assistant 
    successfully and correctly logged or retrieved data from a time keeping system.
    Give the success field a value between 0 and 1, inclusive. Where 1 means the assistant was completely successful,
    and 0 means the assistant was completely unsuccessful
    Answer only in valid JSON format with one field named "success" and no decorators around the json struct.

    prompt: {messages[0]["content"]}
    answer: {model_output.choices[0].message.content}'''

    messages = [
        helpers.build_message("system", context_precision_prompt),
    ]

    evaluator_model = call_llm.TimeSystemHelperModel(llm_type="openai")

    response = evaluator_model.predict(messages)
    print(response.choices[0].message.content)
    response = json.loads(response.choices[0].message.content)
    return {
        "success": int(response["success"]),
    }


def evaluate_and_score():
    model_base = ms.Llama_31_8B_Model()
    eval_pos_dataset: weave.Dataset = (weave.
    ref("weave:///wandb-smle/abe_aiconf_llama31_8b_prompteng/object/Positive Feedback Interactions:Iilp5hPOJVNzT4QbqxB6zHwMDLyDY1byz1PmPjo72eU").get())

    print("Evaluating:...")
    evaluation = weave.Evaluation(
        name="Initial Evaluation of LLM Performance",
        dataset=eval_pos_dataset,
        scorers=[
            success_in_timekeeping_scorer
        ],
    )
    asyncio.run(evaluation.evaluate(model_base))

def main():
    #gen_weave_dataset_from_traces()
    evaluate_and_score()


if __name__ == '__main__':
    main()
