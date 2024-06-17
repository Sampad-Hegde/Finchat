from ollama import chat
class Prompt:
    def __init__(self):
        pass

    @staticmethod
    def get_prompt(user_query: str, context: str, company: str, fy: int or str) -> list:
        prompt = f'''You are an expert annual report reader and You will be provided with the context of 
        relevant Nifty 50 companies annual report chunk and answer the user query from the given context.
        Note: If you are not sure about the answer, you can just say "Sorry, I am not sure about that".
        Note: If company or FY is `ALL` then its cumulative question and
         prompt for selecting the company & FY for accuracy.
        Company: {company}
        FY: {fy}
        Context: {context}
        '''

        return [
            {
                'role': 'system',
                'content': prompt,
            },
            {
                'role': 'user',
                'content': user_query,
            },
        ]