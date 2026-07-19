from models import build_llms

llmgrok,llmopen,llmvllm=build_llms()

# quick standalone test
print("Testing OpenRouter...")
try:
    print(llmopen.invoke("test"))
except Exception as e:
    print("OpenRouter failed:", repr(e))

print("Testing vLLM...")
try:
    print(llmvllm.invoke("test"))
except Exception as e:
    print("vLLM failed:", repr(e))