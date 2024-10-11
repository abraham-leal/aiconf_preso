import weave
from dotenv import load_dotenv
from datasets import load_dataset
import model_service as ms
import feedback_service as fs
import evaluator as eval
import prompt_impv_service as pis

load_dotenv()

client = weave.init("wandb-smle/WB_Weave_NVIDIA_NIMs_DEMO")


def main():
    base_prompt = """
                    You are an assistant that answers questions.
                    """
    for i in range(5):
        print(f"base prompt: {base_prompt}")
        base_prompt = orchestrate(base_prompt, itr_samples=10)

@weave.op()
def orchestrate(base_prompt: str, itr_samples: int):
    model = ms.NIM_Llama3(context=base_prompt)
    model31 = ms.NIM_Llama31(context=base_prompt)
    feedback_mod = fs.Feedback_Service_Model_GPT4oMini()
    prompt_imp_mod = pis.PromptImp_Servicel_GPT4o()

    ds = load_dataset("mosaicml/dolly_hhrlhf", split="test")
    suffled_ds = ds.shuffle()
    iter = suffled_ds.iter(batch_size=1)
    counter = 0
    for obj in iter:
        counter += 1
        start = 'Instruction:\n'
        end = '\n\n### Response:\n'
        s = obj["prompt"][0]
        prompt = s[s.find(start) + len(start):s.rfind(end)]
        label = obj["response"][0]

        messages = build_message("user", prompt)
        completion, orig_call = model.predict.call(model, messages)
        completion31, orig_call31 = model31.predict.call(model31, messages)

        msg_for_feedback = build_message("user", f"""
        prompt: {prompt}
        ideal_answer: {label}
        answer: {completion.choices[0].message.content}
            """)
        msg_for_feedback31 = build_message("user", f"""
                prompt: {prompt}
                ideal_answer: {label}
                answer: {completion31.choices[0].message.content}
                    """)

        feedback_completion = feedback_mod.predict(msg_for_feedback)
        feedback_completion31 = feedback_mod.predict(msg_for_feedback31)


        if "👍" in feedback_completion.choices[0].message.content:
            orig_call.feedback.add_reaction("👍")
        else:
            orig_call.feedback.add_reaction("👎")

        if "👍" in feedback_completion31.choices[0].message.content:
            orig_call31.feedback.add_reaction("👍")
        else:
            orig_call31.feedback.add_reaction("👎")

        if counter >= itr_samples:
            break

    pos_dataset, neg_dataset = eval.gen_weave_dataset_from_traces(client)
    eval.evaluate_and_score(model, pos_dataset, neg_dataset)
    eval.evaluate_and_score(model31, pos_dataset, neg_dataset)

    return prompt_imp_mod.predict(build_message("user", base_prompt)).choices[0].message.content


def build_message(type: str, message):
    return {"role": f'{type}', "content": f'{message}'}


if __name__ == '__main__':
    main()
