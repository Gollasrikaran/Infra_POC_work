import google.generativeai as genai

genai.configure(api_key="")
for m in genai.list_models():
    if 'gemini-3' in m.name:
        print(m.name)
