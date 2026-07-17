import sys, json, re, io
sys.path.insert(0, '.')
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from evalplus.data import get_human_eval_plus

SIMPLE_PROMPT = "### Instruction:\n{instruction}\n### Response:\n"

model_path = './outputs/code_sft/final'
tok = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
).cuda().eval()

problems = get_human_eval_plus()
path = '/tmp/humaneval_simple.jsonl'
with open(path, 'w') as f:
    for task_id, p in tqdm(problems.items(), desc='HumanEval (simple prompt)'):
        prompt = SIMPLE_PROMPT.format(instruction=p['prompt'])
        inp = tok(prompt, return_tensors='pt').to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inp,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        text = tok.decode(
            out[0][inp['input_ids'].shape[1]:],
            skip_special_tokens=True,
        )
        code = re.findall(
            r'```(?:python)?\s*(.*?)```',
            text,
            re.DOTALL,
        )
        f.write(
            json.dumps({
                'task_id': task_id,
                'completion': code[0].strip() if code else text.strip(),
            }) + '\n'
        )

print("Evaluating...")
from evalplus.evaluate import evaluate
buf = io.StringIO()
with __import__('contextlib').redirect_stdout(buf):
    evaluate(samples=path, dataset='humaneval', parallel=4)
print(buf.getvalue())