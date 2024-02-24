import openllm

client = openllm.client.HTTPClient('http://localhost:3000')
client.query({"message":"Explain to me the difference between further and farther"})
