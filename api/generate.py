from transformers import T5Tokenizer, AutoModelForCausalLM
import torch
# トークナイザーとモデルの準備
tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt-1b")
model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt-1b")

# 続きを生成したいテキスト
text = "櫨山研究室は，ソフトウェア工学の研究室です．"


# テキストのエンコード
token_ids = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")

# 文章生成
with torch.no_grad():
    output_ids = model.generate(
        token_ids,
        do_sample=True,
        max_length=100,
        min_length=100,
        top_k=500,
        top_p=0.95,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bad_word_ids=[[tokenizer.unk_token_id]]
    )

# テキストのデコード
sentences = tokenizer.batch_decode(output_ids.tolist())
for sentence in sentences:
    print(sentence)