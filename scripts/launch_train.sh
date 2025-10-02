UV_ENV_FILE=.env CUDA_VISIBLE_DEVICES=0 uv run accelerate launch --multi_gpu -m fineweb_vi_llm.training.train \
    outputs/date-251109 \
    --tokenized-data-uri thng292/fw-experiment-1-tokenized-packed \
    --checkpoint-dir outputs/training-251109 \
    --checkpoint-step 0.1 \
    --eval-step 0.1 \
    --batch-size-per-device 2 \
    --gradient-accumulation 1 \
    --eval-batch-size-per-device 1 \
    --epochs 0.01 \
    --learning-rate 1e-5 \
    # --train-float16 \
    --debug \
    --use-adamw \
    --no-eval-on-start 