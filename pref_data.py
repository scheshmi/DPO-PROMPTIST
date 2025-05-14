import pandas as pd


df_original = pd.read_csv("human_preferences-100k.csv")


df_new = pd.DataFrame(columns=['prompt', 'accepted', 'rejected'])


for index, row in df_original.iterrows():
    if (index%1000==0):
        print(index)
    prompt = row['prompt']
    generated_prompt1 = row['generated_prompt1']
    generated_prompt2 = row['generated_prompt2']
    prompt1_score = row['prompt1_score']
    prompt2_score = row['prompt2_score']

    if prompt1_score > prompt2_score:
        accepted = generated_prompt1
        rejected = generated_prompt2
    else:
        accepted = generated_prompt2
        rejected = generated_prompt1
    
    df_new = pd.concat([df_new, pd.DataFrame({'prompt': [prompt], 'accepted': [accepted], 'rejected': [rejected]})], ignore_index=True)

df_new.to_csv("accepted_rejected.csv", index=False)
