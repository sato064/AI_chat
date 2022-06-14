from transformers import T5Tokenizer, AutoModelForCausalLM
import torch

from collections import defaultdict

text = input()
retext = ''

hash = lambda: defaultdict(hash)
dic = hash()
dic['学芸大丼'] = '学芸大丼は量が多いです….'
dic['櫨山研究室'] = '櫨山研究室はソフトウェア開発をやっています.'

bi_hash = lambda: defaultdict(bi_hash)
bi_dic = bi_hash()
bi_dic['学芸大']['特徴'] = '学芸大は，こんな名前だけど国立大学なんですよ． 実は武蔵小金井のほうが近いんですよね． こんなやつが教壇に立って良いのかと思うやつが多々いるって誰かが言ってました． 実は学芸大学駅から遠いのですよ．'
bi_dic['学芸大']['体育'] = '学芸大の体育だとボウリングが楽しいって噂ですよね．'
bi_dic['学芸大']['学務課'] = '学芸大の学務課は閉まるのが早いですよね． たしか16:15とか．'
bi_dic['学芸大']['守衛'] = '学芸大って学生証なくても入れます？'
bi_dic['学芸大']['駅'] = '学芸大って駅から遠いですよね…'
bi_dic['学芸大']['オタク'] = '学芸大のオタクはしぶんけんに集まってます．'
bi_dic['学食']['おすすめ'] = '学芸大の学食のおすすめは味噌バターコーンラーメンですっ！あと，カレーもおいしいですよね．'

tri_hash = lambda: defaultdict(tri_hash)
tri_dic = tri_hash()
tri_dic['学芸大']['おすすめ']['場所'] = '学芸大で好きな場所と言えば…農場です．'
tri_dic['学芸大']['建物']['高い'] = '学芸で高い建物は，サンシャインですよねっ！'

for word in dic:
    if word in text:
        # print('確認', dic.get(word))
        retext = retext + dic.get(word)

for word in bi_dic:
    for sub_word in bi_dic[word]:
        if word in text and sub_word in text:
            # print('確認', bi_dic[word].get(sub_word))
            retext = retext + bi_dic[word].get(sub_word)

for word in tri_dic:
    for sub_word in tri_dic[word]:
        for subsub_word in tri_dic[word][sub_word]:
            if word in text and sub_word in text and subsub_word in text:
                # print('確認', tri_dic[word][sub_word].get(subsub_word))
                retext = retext + tri_dic[word][sub_word].get(subsub_word)

if not retext:
    retext = text
    # print('確認 retextは', retext)

# トークナイザーとモデルの準備
tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt-1b")
model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt-1b")

# 続きを生成したいテキスト
text = retext

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