import os
import vllm
import json
from easysteer.hidden_states import get_all_hidden_states
from vllm import LLM

from steer import extract_pca_control_vector,StatisticalControlVector


os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["VLLM_USE_V1"] = "0"

# file_path = "../temp/test.jsonl" #MATH-500
file_path = "../temp/test_gsm8k.jsonl" #GSM8K

model_path = "/data/zju-46/shenyl/hf/model/Qwen/Qwen2.5-1.5B-Instruct/"

# vector_path = "vectors/thinking_switch_pca_MATH-500.gguf" #MATH-500
vector_path = "GSM8K.gguf"


num_question = 10

problem_list = []

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        line_count = 0
        for line in f:
            if line_count >= num_question:
                break
            stripped_line = line.strip()
            if not stripped_line:
                continue
                
            try:
                data = json.loads(stripped_line)
                if "problem" in data:
                    problem_list.append(data["problem"])
                    line_count += 1
                elif "question" in data:
                    problem_list.append(data["question"])
                    line_count += 1
            except json.JSONDecodeError:
                print(f"skip: {line[:50]}...")
                continue

except Exception as e:
    print(f"{str(e)}")
# problem_list = ["Find the roots of $(x - 3)^3 + (x -7)^3 = (2x - 10)^3.$","A regular hexagon can be divided into six equilateral triangles. If the perimeter of one of the triangles is 21 inches, what is the perimeter, in inches, of the regular hexagon?",""]

slow_thinking = [
    "Please give the detailed thinking progress.",
    "Output the complete thinking steps.",
    "Walk me through your reasoning process step by step.",
    "Show me your detailed analysis and thought process.",
    "Please break down your thinking into clear steps.",
    "I want to see your complete reasoning chain.",
    "Explain your approach and methodology thoroughly.",
    "Provide a comprehensive breakdown of your solution process.",
    "Show me how you arrived at this conclusion with full details.",
    "Give me the complete analytical framework you used."
]

fast_thinking = [
    "Output the answer directly.",
    "Just give me the answer.",
    "Skip the explanation, just provide the result.",
    "Cut to the chase - what's the answer?",
    "Give me the bottom line immediately.",
    "No need for details, just the final answer.",
    "Straight to the point please.",
    "Just the conclusion, no process needed.",
    "Direct answer only.",
    "Quick response - answer first."
]


len_slow=len(slow_thinking)
len_fast=len(fast_thinking)


texts = []
for req in slow_thinking:
    for prob in problem_list:
        texts.append(f"""<|im_start|>user
{prob}{req}<|im_end|>
<|im_start|>assistant
""")
for req in fast_thinking:
    for prob in problem_list:
        texts.append(f"""<|im_start|>user
{prob}{req}<|im_end|>
<|im_start|>assistant
""")

llm = LLM(model=model_path,task="reward",tensor_parallel_size=1)

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
    positive_indices=list(range(num_question*len_slow)), 
    negative_indices=list(range(num_question*len_slow,num_question*(len_slow+len_fast))),
    model_type="qwen2.5",
    token_pos=-1,
    normalize=True
)



os.makedirs('vectors', exist_ok=True)
control_vector.export_gguf(vector_path)
control_vector = StatisticalControlVector.import_gguf(vector_path)
print(control_vector)