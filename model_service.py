import weave
import os
from dotenv import load_dotenv
from openai import OpenAI
import orchestrator

load_dotenv()

class Llama_31_8B_Model(weave.Model):

    @weave.op()
    def _call_openrouter(self, model, messages):
        or_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OR_API_KEY"))

        try:
            completion = or_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
            )
        except Exception as e:
            print("Unable to generate ChatCompletion response")
            print(f"Exception: {e}")
            return e

        return completion

    @weave.op()
    def call_llama(self, messages):
        return self._call_openrouter("meta-llama/llama-3.1-8b-instruct:free", messages)

    @weave.op()
    def predict(self, messages: []):
        context = """
                    You are a general purpose assistant. Answer the following question to the best of your knowledge.
                    """

        sys_messages = [orchestrator.build_message("system", context), messages]

        return self.call_llama(sys_messages)

