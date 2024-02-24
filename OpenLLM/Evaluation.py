import torch
from transformers import AutoModelForCausalLM

model_name = "microsoft/phi-2"
model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
text = "Sample text for benchmarking"
input_ids = model.tokenizer(text, return_tensors="pt").input_ids.cuda()
reps =100
times = []

for i in range(reps):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # Start timer
    start.record()
    # Model inference
    outputs = model(input_ids).logits
    # End timer
    end.record()
    # Sync and get time
    torch.cuda.synchronize()
    times.append(start.elapsed_time(end))

# Calculate TPS
tokens = len(text.split())
tps = (tokens * reps) / sum(times)
# Calculate latency
latency = sum(times) / reps * 1000 # in ms
print(f"Avg TPS: {tps:.2f}")
print(f"Avg Latency: {latency:.2f} ms")