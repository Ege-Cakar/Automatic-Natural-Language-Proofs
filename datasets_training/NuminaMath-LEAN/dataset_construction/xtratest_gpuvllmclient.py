from openai import OpenAI

#sq
SERVER_NODE="localhost"
SERVER_PORT=8000
openai_api_key = "EMPTY"
openai_api_base = f"http://{SERVER_NODE}:{SERVER_PORT}/v1/"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

chat_response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-Prover-V2-7B",
    messages=[
        {"role": "user", "content": "Hi there!"},
    ],
    max_tokens=10000,
    temperature=0.6,
    top_p=0.95
)
modelOut = chat_response.choices[0].message.content
modelOut = modelOut if modelOut is not None else "[No output returned]"
print (modelOut)