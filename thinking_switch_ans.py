import os
import datetime
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


from vllm.steer_vectors.request import SteerVectorRequest
from vllm import LLM, SamplingParams


from steer import StatisticalControlVector

model_name = "/data/zju-46/shenyl/hf/model/Qwen/Qwen2.5-1.5B-Instruct/"
vector_path="/home/meixinyu/EasySteer/temp/thinking_switch_pca_vector.gguf" #change to your path

control_vector = StatisticalControlVector.import_gguf(vector_path)
print(control_vector)

steer_vector_request_pos = SteerVectorRequest(
            steer_vector_name="thinking_switch_pos",
            steer_vector_id=1,
            steer_vector_local_path=vector_path,
            scale=10.0,
            prefill_trigger_tokens="-1",
            prefill_trigger_positions=[-1],
            generate_trigger_tokens="-1",
            debug=True,
            algorithm='direct'
        )
steer_vector_request_neg = SteerVectorRequest(
            steer_vector_name="thinking_switch_neg",
            steer_vector_id=2,
            steer_vector_local_path=vector_path,
            scale=-10.0,
            prefill_trigger_tokens="-1",
            prefill_trigger_positions=[-1],
            generate_trigger_tokens="-1",
            debug=True,
            algorithm='direct'
        )
sampling_params = SamplingParams(temperature=0.0,max_tokens=1280)
prompt_template = "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n"
prompt="There are 32 dogs and chickens in the cage, and there are 56 legs in it. How many dogs and chickens?"
input = prompt_template % prompt
llm = LLM(model=model_name, enable_steer_vector=True, tensor_parallel_size=1)
output_pos = llm.generate(
                input,
                sampling_params,
                steer_vector_request=steer_vector_request_pos
            )
generated_text_pos = output_pos[0].outputs[0].text

output_neg = llm.generate(
                input,
                sampling_params,
                steer_vector_request=steer_vector_request_neg
            )
generated_text_neg = output_neg[0].outputs[0].text

print(generated_text_pos+"\n\n")
print(generated_text_neg)

os.makedirs('temp', exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = os.path.join('temp', f'thinking_switch_{timestamp}.txt')

with open(filename, 'w', encoding='utf-8') as f:
    f.write("=== Positive: Slow thinking ===\n")
    f.write(generated_text_pos + "\n\n")
    f.write("=== Negative: Fast thinking ===\n")
    f.write(generated_text_neg + "\n")