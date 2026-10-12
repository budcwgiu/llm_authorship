"""
Builds data.csv (llm, input, output, task) from lmsys/chatbot_arena_conversations.
pip install datasets pandas
You may need to accept the dataset terms on its Hugging Face page and run `huggingface-cli login`.
"""
import re
import pandas as pd
from datasets import load_dataset

# Map raw model names to families. Check the printed counts and edit this to fit.
FAMILY = {
    "gpt-4": "gpt", "gpt-3.5-turbo": "gpt",
    "claude-v1": "claude", "claude-instant-v1": "claude", "claude-2.0": "claude", "claude-2.1": "claude",
    "vicuna-13b": "llama", "vicuna-7b": "llama", "vicuna-33b": "llama", "llama-13b": "llama",
    "koala-13b": "llama", "alpaca-13b": "llama", "llama-2-13b-chat": "llama", "llama-2-70b-chat": "llama",
    "palm-2": "palm",
}

def task_of(p):
    p = p.lower()
    if re.search(r"\b(python|code|function|javascript|java|c\+\+|sql|bug|regex|html|css|script)\b", p): return "coding"
    if re.search(r"\b(solve|equation|calculate|math|probability|integral|derivative|how many|sum of)\b", p): return "math"
    if re.search(r"\b(write|story|poem|essay|email|letter|rewrite|song|script for)\b", p): return "writing"
    return "qa"

ds = load_dataset("lmsys/chatbot_arena_conversations", split="train")
rows = []
for r in ds:
    if r.get("language") != "English": continue
    for m, conv in ((r["model_a"], r["conversation_a"]), (r["model_b"], r["conversation_b"])):
        fam = FAMILY.get(m)
        if not fam or len(conv) < 2: continue
        prompt, resp = conv[0]["content"], conv[1]["content"]
        if len(resp.split()) < 5: continue
        rows.append((fam, prompt, resp, task_of(prompt), r["question_id"]))

df = pd.DataFrame(rows, columns=["llm", "input", "output", "task", "qid"])
print(df.llm.value_counts()); print(df.task.value_counts())
df.drop(columns="qid").to_csv("data.csv", index=False)
print("saved data.csv", len(df))
