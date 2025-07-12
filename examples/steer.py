import os
import datetime
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


from vllm.steer_vectors.request import SteerVectorRequest
from vllm import LLM, SamplingParams


from easysteer.steer import StatisticalControlVector

model_path = "/data/zju-46/shenyl/hf/model/Qwen/Qwen2.5-1.5B-Instruct/"
# vector_path="/home/meixinyu/EasySteer/vectors/thinking_switch_pca_MATH-500.gguf" #change to your path
vector_path="/home/meixinyu/EasySteer/vectors/thinking_switch_pca_GSM8K.gguf" #change to your path

control_vector = StatisticalControlVector.import_gguf(vector_path)
print(control_vector)

steer_vector_request_pos = SteerVectorRequest(
            steer_vector_name="thinking_switch_pos",
            steer_vector_id=1,
            steer_vector_local_path=vector_path,
            scale=16.0,
            prefill_trigger_tokens="-1",
            prefill_trigger_positions=[-1],
            generate_trigger_tokens="-1",
            debug=False,
            algorithm='direct'
        )
steer_vector_request_neg = SteerVectorRequest(
            steer_vector_name="thinking_switch_neg",
            steer_vector_id=2,
            steer_vector_local_path=vector_path,
            scale=-16.0,
            prefill_trigger_tokens="-1",
            prefill_trigger_positions=[-1],
            generate_trigger_tokens="-1",
            debug=False,
            algorithm='direct'
        )
sampling_params = SamplingParams(temperature=0.0,max_tokens=1280)
prompt_template = "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n"
# prompt = "Find the constant term in the expansion of $$\\left(10x^3-\\frac{1}{2x^2}\\right)^{5}$$"
prompt = "Darrell and Allen's ages are in the ratio of 7:11. If their total age now is 162, calculate Allen's age 10 years from now."
input = prompt_template % prompt
llm = LLM(model=model_path, enable_steer_vector=True, tensor_parallel_size=1)
output_base = llm.generate(
    input,
    sampling_params
)
generated_text_base = output_base[0].outputs[0].text

llm = LLM(model=model_path, enable_steer_vector=True, tensor_parallel_size=1)
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

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f'thinking_switch_GSM8K_{timestamp}.txt'

with open(filename, 'w', encoding='utf-8') as f:
    f.write("=== Base ===\n")
    f.write(generated_text_base + "\n\n")
    f.write("=== Positive: Slow thinking ===\n")
    f.write(generated_text_pos + "\n\n")
    f.write("=== Negative: Fast thinking ===\n")
    f.write(generated_text_neg + "\n")