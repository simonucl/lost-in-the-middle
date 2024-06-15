for gold_index in 0 4 9 14 19; do
    python -u ./scripts/make_qa_data_from_retrieval_results.py \
        --input-path nq-open-contriever-msmarco-retrieved-documents.jsonl.gz \
        --num-total-documents 20 \
        --gold-index ${gold_index} \
        --output-path qa_data/nq-open-20_total_documents_gold_at_${gold_index}.jsonl.gz
done