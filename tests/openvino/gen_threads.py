from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer, AutoConfig, LlamaTokenizer
import threading

model_path = "/home/devuser/openvino.genai/llm_bench/python/llama-2-7b-chat-hf-stateful/pytorch/dldt/compressed_weights/OV_FP16-INT8/"
model_path = "/home/devuser/openvino.genai/llm_bench/python/llama-2-7b-chat-hf/pytorch/dldt/compressed_weights/OV_FP16-INT8/"
tokenizer = LlamaTokenizer.from_pretrained(model_path)
tokenizer.pad_token = "[PAD]"
tokenizer.padding_side = "left"

prompt1 = [" The weather is "]
prompt2 = [" Openvino is a ", "The relativity theory is created "]
#prompt3 = [" Are cats smarter that dogs ", " How big is an elephant ", " How small could be a red ant "]
prompt3 = [" Are cats smarter that dogs ", " How big is an elephant ", " the water in the ocean is much hotter than before  "]


OV_CONFIG = {'PERFORMANCE_HINT': 'LATENCY', 'CACHE_DIR': '','NUM_STREAMS': '1'}
model = OVModelForCausalLM.from_pretrained(model_path, config=AutoConfig.from_pretrained(model_path, trust_remote_code=True),ov_config=OV_CONFIG)


results = [None]*3

def gen_thread(prompt, results, i):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    generate_kwargs = dict(
            input_ids=inputs.input_ids,
            max_new_tokens=50,
            temperature=1.0,
            do_sample=True,
            top_p=1.0,
            top_k=50,
            repetition_penalty=1.1
             )
    outputs = model.generate(**generate_kwargs)
    results[i] = outputs


x = threading.Thread(target=gen_thread, args=(prompt1,results, 0))
x.start()
y = threading.Thread(target=gen_thread, args=(prompt2,results, 1))
y.start()
z = threading.Thread(target=gen_thread, args=(prompt3,results, 2))
z.start()
x.join()
y.join()
z.join()

def print_response(res):
    for answer in res:
        print("Answer:")
        print(tokenizer.decode(answer, skip_special_tokens=True))

print("THREAD1")
print(prompt1)
print_response(results[0])

print("THREAD2")
print(prompt2)
print_response(results[1])

print("THREAD3")
print(prompt3)
print_response(results[2])



