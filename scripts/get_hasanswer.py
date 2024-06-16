# load the first 100 rows from "nq-open-contriever-msmarco-retrieved-documents.jsonl"
from tqdm import tqdm
import json
docs = []
with open('nq-open-contriever-msmarco-retrieved-documents.jsonl', 'r') as f:
    for i, line in enumerate(tqdm(f)):
        docs.append(json.loads(line))

noanswer = []
for i, doc in enumerate(docs):
    hasanswer = [d['hasanswer'] for d in doc['ctxs']]
    # if all false, print(i)
    if not any(hasanswer):
        noanswer.append(i)

new_docs = []
for i in range(len(docs)):
    hasanswer = [d['hasanswer'] for d in docs[i]['ctxs']]
    if not any(hasanswer):
        continue
    # get the first context with hasanswer == True
    ctx = docs[i]['ctxs'][hasanswer.index(True)]
    new_docs.append({
        'question': docs[i]['question'],
        'gold_id': ctx['id'],
        'answers': docs[i]['answers'],
        'ctxs': [{'title': ctx['title'], 'text': ctx['text'], 'isgold': True, 'hasanswer': True}],
        'nq_annotated_gold': {'title': ctx['title'], 'long_answer': ctx['text'], 'chunked_long_answer': ctx['text']},
        'original_nq_annotated': docs[i]['nq_annotated_gold']
    })

# save the new docs as nq-open-oracle-doc.jsonl
with open('nq-open-oracle-doc.jsonl', 'w') as f:
    for doc in new_docs:
        f.write(json.dumps(doc) + '\n')