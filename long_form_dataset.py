import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os

# 1. Setup
os.environ['CUDA_VISIBLE_DEVICES'] = '3,1'
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_FILE = "comprehensive_metric_test.csv"
CHECKPOINT_INTERVAL = 5 # Save progress every 5 rows

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto"
)

ASPECTS = {
    "Semantic Depth": "Rewrite to be more nuanced and detailed without changing the facts.",
    "Synonyms": "Replace key words with synonyms while keeping meaning identical.",
    "Morphology": "Change tenses or word forms (e.g., singular to plural) while keeping meaning.",
    "Negation": "Flip the meaning to the opposite (add 'not', 'never', etc.).",
    "Paraphrasing": "Completely rewrite the structure while keeping the same meaning.",
    "Reordering": "Change the sequence of sentences or clauses without changing the meaning.",
    "Passive Voice": "Convert all active sentences to passive voice where possible.",
    "Indirect Speech": "Convert direct statements into reported/indirect speech.",
    "Numbers": "Slightly alter numerical values (e.g., 100 to 101) or change their format."
}

def generate_variation(aspect_name, instruction, text):
    messages = [
        {"role": "system", "content": f"You are a linguistic expert. Perform the following transformation: {instruction}. Return ONLY the transformed text. Do not explain your changes."},
        {"role": "user", "content": f"Text to transform: {text}"}
    ]
    
    # 1. Get both input_ids and attention_mask
    inputs = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        return_tensors="pt",
        return_dict=True  # This is the key change
    ).to(DEVICE)
    
    input_length = inputs['input_ids'].shape[1]
    max_gen = min(2048, 4096 - input_length) 

    with torch.no_grad():
        # 2. Pass the dictionary unpack (**inputs) which includes the mask
        output_ids = model.generate(
            **inputs, 
            max_new_tokens=max_gen,
            temperature=0.2,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id  # Explicitly set this here too
        )
    
    response = tokenizer.decode(output_ids[0][input_length:], skip_special_tokens=True)
    return response.strip()

# 3. Expansion Logic with Advanced Tracking
def create_expanded_dataset(df):
    results = []
    
    # Outer progress bar for original rows
    pbar_rows = tqdm(df.iterrows(), total=len(df), desc="Total Progress", position=0)
    
    for i, (_, row) in enumerate(pbar_rows):
        # A. Store the Control
        control_row = row.to_dict()
        control_row['Aspect'] = 'Control'
        results.append(control_row)
        
        # B. Inner progress bar for the 9 variations
        pbar_aspects = tqdm(ASPECTS.items(), desc=f"Row {i} Aspects", leave=False, position=1)
        
        for aspect, instr in pbar_aspects:
            try:
                perturbed_text = generate_variation(aspect, instr, row['answer'])
                new_row = row.to_dict()
                new_row['Response'] = perturbed_text
                new_row['Aspect'] = aspect
                results.append(new_row)
            except Exception as e:
                print(f"\nError on Row {i}, Aspect {aspect}: {e}")
                continue
        
        # 4. Checkpointing: Save every N rows
        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
            
    return pd.DataFrame(results)

# 4. Run the expansion
df = pd.read_csv('Input_data/Testing_dataset.csv')
# Start with a small subset to verify timing, e.g., df.head(5)
expanded_df = create_expanded_dataset(df)
expanded_df.to_csv(OUTPUT_FILE, index=False)
print(f"Done! Final dataset saved to {OUTPUT_FILE}")