import weave
import os
import orchestrator
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class PromptImp_Servicel_GPT4o(weave.Model):

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
        context = f"""
                        You are an LLM Prompt improvement Service.
                        You will be given a prompt by the user that an LLM is currently using for answering questions
                        and you will output a new prompt
                        that should make the LLM better at general reasoning tasks, more clear and more concise in its answer.
                        The new prompt should be as detailed as possible.
                        """

        sys_messages = [orchestrator.build_message("system", context), messages]

        return self.call_openai(sys_messages)
