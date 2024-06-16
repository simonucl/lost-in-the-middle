from huggingface_hub import hf_hub_download
import datasets
import pandas as pd
from tqdm import tqdm
import tiktoken
import os
import multiprocessing

# Create a title folder
os.makedirs("titles", exist_ok=True)
os.makedirs("segments", exist_ok=True)

# Initialize the tokenizer
enc = tiktoken.get_encoding('cl100k_base')

# Create an array from 00000 to 00156
wiki_dpr_ids = [f"{str(i).zfill(5)}" for i in range(157)]

def process_wiki_dpr_id(wiki_dpr_id):
    # Download the dataset from Hugging Face hub
    hf_hub_download(repo_id="facebook/wiki_dpr", filename=f"data/psgs_w100/nq/train-{wiki_dpr_id}-of-00157.parquet", repo_type="dataset", local_dir="cache")

    df = pd.read_parquet(f"cache/data/psgs_w100/nq/train-{wiki_dpr_id}-of-00157.parquet")
    df['length'] = df['text'].apply(enc.encode).apply(len)

    # for max_length in [2048, 4096, -1]:
    for max_length in [-1]:
        title2id = {}
        title = ""
        ids = []
        segments = []
        segment_texts = []
        segment_ids = []
        length = 0
        for idx, row in df.iterrows():
            if row['title'] != title:
                # If the title is not empty, append the title, ids, and segment_texts to the respective lists
                if title != "":
                    title2id[title] = ids
                    segments.append(
                        {
                            'title': title,
                            'ids': segment_ids,
                            'texts': " ".join(segment_texts),
                            'length': length
                        }
                    )
                # Reset the title, ids, and segment_texts
                title = row['title']
                ids = []
                segment_texts = []
                segment_ids = []
                length = 0

            ids.append(row['id'])
            if max_length != -1:
                if length + row['length'] > max_length:
                    segments.append(
                        {
                            'title': title,
                            'ids': segment_ids,
                            'texts': " ".join(segment_texts),
                            'length': length
                        }
                    )
                    segment_texts = []
                    segment_ids = []
                    length = 0

            segment_texts.append(row['text'])
            segment_ids.append(row['id'])
            length += row['length']

        if title != "":
            title2id[title] = ids
            segments.append(
                {
                    'title': title,
                    'ids': segment_ids,
                    'texts': " ".join(segment_texts),
                    'length': length
                }
            )

        # Store the segments in a json file
        pd.DataFrame(title2id.items(), columns=['title', 'ids']).to_json(f"titles/{wiki_dpr_id}_title2id.json", orient='records', lines=True)
        os.makedirs(f"segments/{wiki_dpr_id}/{max_length if max_length != -1 else 'all'}", exist_ok=True)
        pd.DataFrame(segments).to_json(f"segments/{wiki_dpr_id}/{max_length if max_length != -1 else 'all'}/segments.json", orient='records', lines=True)

def process_with_tqdm(wiki_dpr_ids):
    cpu_count = 8
    with multiprocessing.Pool(processes=cpu_count) as pool:
        for _ in tqdm(pool.imap_unordered(process_wiki_dpr_id, wiki_dpr_ids), total=len(wiki_dpr_ids)):
            pass

if __name__ == '__main__':
    process_with_tqdm(wiki_dpr_ids)
