from st.models.llama3 import ModelArgs

model_presets = {
    "llama-iwslt": {
        "124m": ModelArgs(n_layers=12, n_heads=12, dim=768, vocab_size=49157, max_seq_len=1024, intermediate_size=4 * 768, n_kv_heads=12),
        "500m": ModelArgs(n_layers=32, n_heads=16, dim=1024, vocab_size=49157, max_seq_len=1024, intermediate_size=3072, n_kv_heads=4),
        "978m": ModelArgs(n_layers=36, n_heads=20, dim=1280, vocab_size=49157, max_seq_len=1024, intermediate_size=5120, n_kv_heads=4),
        "1b": ModelArgs(n_layers=36, n_heads=20, dim=1280, vocab_size=64000, max_seq_len=1024, intermediate_size=5120, n_kv_heads=4),
        "2b": ModelArgs(n_layers=36, n_heads=32, dim=2048, vocab_size=49157, max_seq_len=1024, intermediate_size=5120, n_kv_heads=8),
    },
}
