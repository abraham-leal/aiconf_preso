import weave
import os
import orchestrator
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class Feedback_Service_Model_GPT4oMini(weave.Model):

    @weave.op()
    def call_openai(self, messages):
        oai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        try:
            completion = oai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.0,
            )
        except Exception as e:
            print("Unable to generate ChatCompletion response")
            print(f"Exception: {e}")
            return e

        return completion

    @weave.op()
    def predict(self, messages: []):
        context = """
                        You are a grader of LLM assistant responses.
                        You will be given a prompt, an ideal_answer, and an answer, and you must grade how well the answer
                        resembles the ideal_answer. You may only answer with these two emojis: 👍 or 👎 
                        Answer with 👍 if the answer resembles the ideal_answer really well.
                        Answer with 👎 if the answer resembles the ideal_answer just ok or badly.
                        """

        sys_messages = [orchestrator.build_message("system", context), messages]

        return self.call_openai(sys_messages)