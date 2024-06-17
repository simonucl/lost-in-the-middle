MODELS=(
    Qwen/Qwen2-7B-Instruct
)

# Close book
for model in "${MODELS[@]}"; do
    echo "Running model: $model in closed book setting"
    python3 -u scripts/get_qa_responses.py \
        --input-path qa_data/nq-open-oracle.jsonl.gz \
        --num-gpus 2 \
        --max-new-tokens 100 \
        --closedbook \
        --model $model \
        --add_system_prompt \
        --output-path qa_predictions/nq-open-oracle-${model}-closedbook-predictions.jsonl.gz

    python3 -u scripts/evaluate_qa_responses.py \
    --input-path qa_predictions/nq-open-oracle-${model}-closedbook-predictions.jsonl.gz \
    --output-path qa_predictions/nq-open-oracle-${model}-closedbook-predictions-scored.jsonl.gz

    # echo "Running model: $model in orcale setting"
    # python3 -u ./scripts/get_qa_responses.py \
    # --input-path qa_data/nq-open-oracle.jsonl.gz \
    # --max-new-tokens 100 \
    # --num-gpus 2 \
    # --model $model \
    # --add_system_prompt \
    # --output-path qa_predictions/nq-open-oracle-${model}-oracle-predictions.jsonl.gz

    # python3 -u scripts/evaluate_qa_responses.py \
    #     --input-path qa_predictions/nq-open-oracle-${model}-oracle-predictions.jsonl.gz \
    #     --output-path qa_predictions/nq-open-oracle-${model}-oracle-predictions-scored.jsonl.gz

    # for gold_index in 0 4 9 14 19 24 29; do
    #     echo "Running model: $model in open book setting with gold index: $gold_index"
    #     python3 -u scripts/get_qa_responses.py \
    #         --input-path qa_data/30_total_documents/nq-open-30_total_documents_gold_at_${gold_index}.jsonl.gz \
    #         --num-gpus 2 \
    #         --max-new-tokens 100 \
    #         --model $model \
    #         --add_system_prompt \
    #         --output-path qa_predictions/nq-open-oracle-${model}-openbook-gold-index-${gold_index}-predictions.jsonl.gz

    #     python3 -u scripts/evaluate_qa_responses.py \
    #         --input-path qa_predictions/nq-open-oracle-${model}-openbook-gold-index-${gold_index}-predictions.jsonl.gz \
    #         --output-path qa_predictions/nq-open-oracle-${model}-openbook-gold-index-${gold_index}-predictions-scored.jsonl.gz

    echo "Running model: $model in orcale doc setting"
    python3 -u ./scripts/get_qa_responses.py \
    --input-path qa_data/nq-open-oracle-doc.jsonl.gz \
    --max-new-tokens 100 \
    --num-gpus 2 \
    --model $model \
    --add_system_prompt \
    --output-path qa_predictions/nq-open-oracle-${model}-oracle-doc-predictions.jsonl.gz

    python3 -u scripts/evaluate_qa_responses.py \
        --input-path qa_predictions/nq-open-oracle-${model}-oracle-doc-predictions.jsonl.gz \
        --output-path qa_predictions/nq-open-oracle-${model}-oracle-doc-predictions-scored.jsonl.gz

    for max_length in "2048" "4096" "all"; do
        echo "Running model: $model in open book setting with max length: $max_length"
        python3 -u scripts/get_qa_responses.py \
            --input-path qa_data/nq-open-${max_length}_oracle_docs.jsonl.gz \
            --num-gpus 2 \
            --max-new-tokens 100 \
            --model $model \
            --add_system_prompt \
            --output-path qa_predictions/nq-open-${max_length}-${model}-oracle-doc-predictions.jsonl.gz

        python3 -u scripts/evaluate_qa_responses.py \
            --input-path qa_predictions/nq-open-${max_length}-${model}-oracle-doc-predictions.jsonl.gz \
            --output-path qa_predictions/nq-open-${max_length}-${model}-oracle-doc-predictions-scored.jsonl.gz
    done
done

# for model in "${MODELS[@]}"; do

#     for gold_index in 0 4 9 14 19 24 29; do
#         echo "Running model: $model in open book setting with gold index: $gold_index"
#         python3 -u scripts/get_qa_responses.py \
#             --input-path qa_data/30_total_documents/nq-open-30_2048_total_documents_gold_at_${gold_index}.jsonl.gz \
#             --num-gpus 2 \
#             --max-new-tokens 100 \
#             --model $model \
#             --add_system_prompt \
#             --output-path qa_predictions/nq-open-30_2048-${model}-openbook-gold-index-${gold_index}-predictions.jsonl.gz

#         python3 -u scripts/evaluate_qa_responses.py \
#             --input-path qa_predictions/nq-open-30_2048-${model}-openbook-gold-index-${gold_index}-predictions.jsonl.gz \
#             --output-path qa_predictions/nq-open-30_2048-${model}-openbook-gold-index-${gold_index}-predictions-scored.jsonl.gz
#     done
# done