import weave
from dotenv import load_dotenv
import model_service as ms
import feedback_service as fs
from datasets import load_dataset

load_dotenv()

weave.init("abe_aiconf_llama31_8B_prompteng")
def main():

    model = ms.Llama_31_8B_Model()
    feedback_mod = fs.Feedback_Service_Model_GPT4oMini()

    ds = load_dataset("mosaicml/dolly_hhrlhf", split="train")
    iter = ds.iter(batch_size=1)
    for obj in iter:
        start = 'Instruction:\n'
        end = '\n\n### Response:\n'
        s = obj["prompt"][0]
        prompt = s[s.find(start) + len(start):s.rfind(end)]
        label = obj["response"][0]

        messages = build_message("user", prompt)
        completion, orig_call = model.predict.call(messages)

        msg_for_feedback = build_message("user", f"""
        prompt: {prompt}
        ideal_answer: {label}
        answer: {completion.choices[0].message.content}
            """)

        feedback_completion = feedback_mod.predict(msg_for_feedback)
        print(feedback_completion.choices[0].message.content)

        if "👍" in feedback_completion.choices[0].message.content:
            orig_call.feedback.add_reaction("👍")
        else:
            orig_call.feedback.add_reaction("👎")


def build_message(type: str, message):
    return {"role": f'{type}', "content": f'{message}'}


if __name__ == '__main__':
    main()