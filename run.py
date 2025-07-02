from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda:7"  

model = AutoModelForCausalLM.from_pretrained(
    "models",
    torch_dtype=torch.float16 
).to(device)

tokenizer = AutoTokenizer.from_pretrained("models")

def chat(messages, max_new_tokens=1024, temperature=0.95, top_p=0.7):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

if __name__ == '__main__':
    prompt = "少突胶质细胞瘤的发病机制？"
    messages = [
        {"role": "user", "content": prompt}
    ]
    print(chat(messages))
