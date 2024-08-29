import weave
import os
from dotenv import load_dotenv
from openai import OpenAI
import orchestrator

load_dotenv()

class Llama_31_8B_Model(weave.Model):
    context: str
    @weave.op()
    def _call_oai(self, model, messages):
        oai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        try:
            completion = oai_client.chat.completions.create(
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
    def call_oai(self, messages):
        return self._call_oai("gpt-4o-mini", messages)

    @weave.op()
    def predict(self, messages: []):
        sys_messages = [orchestrator.build_message("system", self.context), messages]

        return self.call_oai(sys_messages)

