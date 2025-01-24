from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import pytrec_eval
import argparse


def read_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def get_response(model, tokenizer, messages):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    output = model.generate(**model_inputs, return_dict_in_generate=True, output_scores=True, max_new_tokens=512)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in
                     zip(model_inputs.input_ids, output['sequences'])]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    prob_scores = output.scores[0][0].cpu().tolist()
    indx_ab = tokenizer(['A', 'B'])['input_ids']
    log_prob = {'A': prob_scores[indx_ab[0][0]], 'B': prob_scores[indx_ab[1][0]],
                'A_AB': prob_scores[indx_ab[0][0]] / (prob_scores[indx_ab[0][0]] + prob_scores[indx_ab[1][0]])}
    return response, log_prob


def get_mes(query, history=None):
    mes = []
    if history is not None and type(history) == list:
        for x in history:
            mes.append({'role': 'user', 'content': x[0]})
            mes.append({'role': 'assistant', 'content': x[1]})
    mes.append({'role': 'user', 'content': query})
    return mes


def get_query_new_format(query, doc):
    templt = '''### reply strategy:
{text0}

Decide whether you can reply with the above reply strategy according to the following conversation content

### conversation content:
{text1}

### Reply to request:
- If you think that you can reply with the reply strategy, please choose option 'A'. If you think that you can not reply with the reply strategy, please choose option 'B'.
Please make a choice from 'A', 'B'.
Your response should use the specified format <'A' or 'B'>, without any additional comments.'''
    return templt.format(text0=doc, text1=query)


def get_recall_ndcg(qrel, run):
    evaluation_scores = pytrec_eval.RelevanceEvaluator(qrel,
                                                       {'recall.1', 'recall.3', 'recall.5', 'recall.10', 'ndcg_cut.10'})
    scores = evaluation_scores.evaluate(run)

    res = {}
    k_v = [1,3,5,10]
    for k in k_v:
        res[f"Recall@{k}"]=0.0
    for query_id in scores.keys():
        for k in k_v:
            res[f"Recall@{k}"] += scores[query_id]["recall_"+str(k)]
    for k in k_v:
        res[f"Recall@{k}"] = round(res[f"Recall@{k}"]/len(scores),5)

    res["NDCG@10"] = 0.0
    for query_id in scores.keys():
        res["NDCG@{10}"] += scores[query_id]["ndcg_cut_10"]
    res["NDCG@{10}"] = round(res["NDCG@{10}"] / len(scores), 5)
    return res



def read_json_data(json_file):
    data = []
    with open(json_file, 'r') as f:
        for x in f:
            d = json.loads(x)
            data.append(d)
    return data

def recall_ndcg(qrel,run):
    eval_md = pytrec_eval.RelevanceEvaluator(qrel,{'recall.1','recall.3','recall.5','recall.10','ndcg_cut.10'})
    scores = eval_md.evaluate(run)

    res = {}
    keys_r = ['Recall@1','Recall@3','Recall@5','Recall@10','NDCG@10']
    scores_reskey = ['recall_1','recall_3','recall_5','recall_10','ndcg_cut_10']
    res.update(dict(zip(keys_r,[0.0 for _ in range(len(keys_r))])))
    for query_i in scores.keys():
        for i,x in enumerate(keys_r):
            res[x] += scores[query_i][scores_reskey[i]]
    for x in res:
        res[x] = round(res[x]/len(scores),5)
    return res



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',type=str,default='/home/xxx/out_model/qwen2_7b_sft/sft_qwen_7b_lr1e6')
    args = parser.parse_args()

    model_name = args.model_path
    data = read_json_data('./data/my_data_test/test_data.json')
    qrel = read_json_data('./data/my_data_test/qrel.json')

    runs = {}
    for d in data:
        query = d['query']
        doc = d['doc']
        q_form = get_query_new_format(query, doc)
        mess = get_mes(q_form)
        model, tokenizer = read_model(model_name)
        res, log_p = get_response(model, tokenizer, mess)
        print(log_p)
        if d['query_id'] not in runs:
            runs[d['query_id']] = {d['doc_id']: log_p['A_AB']}
        else:
            runs[d['query_id']].update({d['doc_id']: log_p['A_AB']})
    print(runs)
    res = recall_ndcg(qrel,runs)
    print(res)
