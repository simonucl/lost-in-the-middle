MODELS=(
    /mnt/nfs/public/hf/models/meta-llama/Meta-Llama-3-8B-Instruct
    mistralai/Mistral-7B-Instruct-v0.3
    microsoft/Phi-3-mini-128k-instruct
    gradientai/Llama-3-8B-Instruct-262k
    gradientai/Llama-3-8B-Instruct-Gradient-1048k
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

    echo "Running model: $model in orcale setting"
    python3 -u ./scripts/get_qa_responses.py \
    --input-path qa_data/nq-open-oracle.jsonl.gz \
    --max-new-tokens 100 \
    --num-gpus 2 \
    --model $model \
    --add_system_prompt \
    --output-path qa_predictions/nq-open-oracle-${model}-oracle-predictions.jsonl.gz

    python3 -u scripts/evaluate_qa_responses.py \
        --input-path qa_predictions/nq-open-oracle-${model}-oracle-predictions.jsonl.gz \
        --output-path qa_predictions/nq-open-oracle-${model}-oracle-predictions-scored.jsonl.gz
done