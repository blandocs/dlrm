coll=100
for trace in "random" "continuous"; do
python dlrm_s_pytorch.py --arch-sparse-feature-size=16 --arch-embedding-size="50000-50000-50000" --arch-mlp-bot="13-64-16" --arch-mlp-top="64-16-4-1" --mini-batch-size=1 --data-size=100 --inference-only --enable-memory-profiling --ouput-memory-traces $trace --qr-flag --qr-collisions $coll

python dlrm_s_pytorch.py --arch-sparse-feature-size=16 --arch-embedding-size="50000-50000-50000" --arch-mlp-bot="13-64-16" --arch-mlp-top="64-16-4-1" --mini-batch-size=1 --data-size=100 --inference-only --enable-memory-profiling --ouput-memory-traces $trace
done

