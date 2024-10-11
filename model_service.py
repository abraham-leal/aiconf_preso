import weave
import os
from dotenv import load_dotenv
from openai import OpenAI
import orchestrator

load_dotenv()

class OpenAI_GPT_4o_Mini(weave.Model):
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

class NIM_Llama3(weave.Model):
    context: str
    @weave.op()
    def call_llama3(self, messages):
        client_nim_llama3 = OpenAI(
            base_url="http://34.46.19.203:8001/v1",
            api_key="xxx"
        )

        try:
            completion = client_nim_llama3.chat.completions.create(
                model="meta/llama3-8b-instruct",
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
        sys_messages = [orchestrator.build_message("system", self.context), messages]

        return self.call_llama3(sys_messages)

class NIM_Llama31(weave.Model):
    context: str
    @weave.op()
    def call_llama31(self, messages):
        client_nim_llama31 = OpenAI(
            base_url="http://34.46.19.203:8000/v1",
            api_key="xxx"
        )

        try:
            completion = client_nim_llama31.chat.completions.create(
                model="meta/llama-3.1-8b-instruct",
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
        sys_messages = [orchestrator.build_message("system", self.context), messages]

        return self.call_llama31(sys_messages)
