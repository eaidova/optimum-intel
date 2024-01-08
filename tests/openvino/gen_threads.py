from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer, AutoConfig
import threading


#model_path = "/home/devuser/openvino.genai/llm_bench/python/mistral-int8-new-stateful/pytorch/dldt/compressed_weights/OV_FP16-INT8/"
#model_path = "/home/devuser/openvino.genai/llm_bench/python/mistral-int8-new/pytorch/dldt/compressed_weights/OV_FP16-INT8"
#model_path = "/home/devuser/openvino.genai/llm_bench/python/llama-2-7b-chat-hf-stateful/pytorch/dldt/compressed_weights/OV_FP16-INT8/"
model_path = "/home/devuser/openvino.genai/llm_bench/python/llama-2-7b-chat-hf-stateful/pytorch/dldt/compressed_weights/OV_FP16-INT8/"
#model_path = "/home/devuser/openvino.genai/llm_bench/python/llama-2-7b-chat-hf/pytorch/dldt/FP16/"


OV_CONFIG = {'PERFORMANCE_HINT': 'LATENCY', 'CACHE_DIR': '','NUM_STREAMS': '1'}
model = OVModelForCausalLM.from_pretrained(model_path, config=AutoConfig.from_pretrained(model_path, trust_remote_code=True),stateful=True,ov_config=OV_CONFIG)
tokenizer = AutoTokenizer.from_pretrained(model_path)

def gen_thread(prompt,inf_req):
    inputs = tokenizer(prompt, return_tensors="pt")
    generate_kwargs = dict(
            input_ids=inputs.input_ids,
            max_new_tokens=200,
            temperature=1.0,
            do_sample=True,
            top_p=1.0,
            top_k=50,
            repetition_penalty=1.1,
            use_cache=False,
            infer_request=inf_req
            )
    outputs = model.generate(**generate_kwargs)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

inf_req1 = model.create_infer_request()
prompt1 = [" The weather is "]
x = threading.Thread(target=gen_thread, args=(prompt1,inf_req1,))
x.start()
inf_req2 = model.create_infer_request()
prompt2 = [" Openvino is a "]
y = threading.Thread(target=gen_thread, args=(prompt2,inf_req2,))
y.start()

x.join()
y.join()




