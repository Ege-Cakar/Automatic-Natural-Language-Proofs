import google.generativeai as genai

# put your API key here
genai.configure(api_key="blabla")

# pick a model
model = genai.GenerativeModel("gemini-1.5-flash")

response = model.generate_content("Write a short poem about AI")

print(response.text)
