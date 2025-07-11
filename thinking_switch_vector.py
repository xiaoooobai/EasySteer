import os
import vllm
from hidden_states import get_all_hidden_states
from vllm import LLM
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["VLLM_USE_V1"] = "0"

model_name = "/data/zju-46/shenyl/hf/model/Qwen/Qwen2.5-1.5B-Instruct/"
llm = LLM(model=model_name, task="reward", tensor_parallel_size=1)
slow_thinking=["Please give the detailed thinking progress.","Output the complete thinking steps."]
fast_thinking=["Output the answer directly.","Just give me the answer."]
texts = []
for t in slow_thinking+fast_thinking:
    texts.append(f"""<|im_start|>user
{t}<|im_end|>
<|im_start|>assistant
That""")
all_hidden_states, outputs = get_all_hidden_states(llm, texts)

from steer import (extract_statistical_control_vector,
    extract_diffmean_control_vector,
    extract_pca_control_vector,
    extract_lat_control_vector,
    extract_gemmascope_sae_diffmean_control_vector,
    extract_linear_probe_control_vector,
    StatisticalControlVector
)

control_vector = extract_pca_control_vector(
    all_hidden_states=all_hidden_states,
    positive_indices=[0, 1], 
    negative_indices=[2, 3],
    model_type="qwen2.5",
    token_pos=-1,
    normalize=True
)

vector_name="temp/thinking_switch_pca_vector.gguf"

os.makedirs('temp', exist_ok=True)
control_vector.export_gguf(vector_name)
control_vector = StatisticalControlVector.import_gguf(vector_name)
print(control_vector)