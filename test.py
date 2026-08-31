import boto3
import json

bedrock = boto3.client(
    "bedrock-runtime",
    region_name="us-east-1"
)

response = bedrock.invoke_model(
    modelId="openai.gpt-oss-120b-1:0",
    body=json.dumps({
        "messages": [
            {
                "role": "user",
                "content": "Can you explain the features of Amazon Bedrock?"
            }
        ],
        "max_tokens": 512,
        "temperature": 0.7
    }),
    contentType="application/json",
    accept="application/json"
)

result = json.loads(response["body"].read())
print(result)