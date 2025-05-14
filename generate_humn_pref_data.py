import pandas as pd
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import os
import ImageReward as RM
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("gpt2")           
model = AutoModelForCausalLM.from_pretrained(              
    "gpt2",
    state_dict=torch.load("sft_gpt.bin"),
).to(device)
model.eval()

model_id = "CompVis/stable-diffusion-v1-4"
output_folder = 'img'
os.makedirs(output_folder, exist_ok=True)

pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True, torch_dtype=torch.float16)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.to(device)

model_rm = RM.load("ImageReward-v1.0").to(device)

input_file = '100k_prompts.txt' 
with open(input_file, 'r') as file:
    prompts = file.readlines()


df = pd.DataFrame(columns=['prompt', 'generated_prompt1', 'generated_prompt2', 'prompt1_score', 'prompt2_score'])

for i, prompt in enumerate(prompts):
    print(i)
    if i <= 75000:
        continue
    generated_outputs = []
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device) 
    for _ in range(2):
        generated_output = model.generate(
            input_ids,
            max_new_tokens=75,
            num_return_sequences=1,
            do_sample=True, 
            top_k=50,  
            top_p=0.95,  
            early_stopping=True,
        )
        generated_text = tokenizer.decode(generated_output[0], skip_special_tokens=True)
        generated_outputs.append(generated_text)
    output_path1 = os.path.join(output_folder, f'{i}1.png')   
    image = pipe(generated_outputs[0], num_inference_steps=10).images[0]
    image.save(output_path1)
    output_path2 = os.path.join(output_folder, f'{i}2.png')    
    image = pipe(generated_outputs[1], num_inference_steps=10).images[0]
    image.save(output_path2)
    rewards = model_rm.score(prompt, [output_path1, output_path2])
    os.remove(output_path1)
    os.remove(output_path2)


    new_row = pd.DataFrame({
        'prompt': [prompt],
        'generated_prompt1': [generated_outputs[0]],
        'generated_prompt2': [generated_outputs[1]],
        'prompt1_score': [rewards[0]],
        'prompt2_score': [rewards[1]]
    })

    df = pd.concat([df, new_row], ignore_index=True)


    df.to_csv('human_preferences-100k.csv', sep=',', index=False)