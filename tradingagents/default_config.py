import os

DEFAULT_CONFIG = {
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "data_dir": "/Users/yluo/Documents/Code/ScAI/FR1-data",
    "data_cache_dir": os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
        "dataflows/data_cache",
    ),
    # LLM settings - Updated to use HuggingFace model
    "deep_think_llm": "HuggingFaceH4/zephyr-7b-beta",
    "quick_think_llm": "HuggingFaceH4/zephyr-7b-beta",
    # Debate and discussion settings
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    "max_recur_limit": 100,
    # Tool settings
    "online_tools": True,
    # HuggingFace specific settings
    "hf_model_kwargs": {
        "temperature": 0.1,
        "max_length": 2048,
        "do_sample": True,
        "top_p": 0.95,
        "top_k": 50,
    },
    "hf_pipeline_kwargs": {
        "device_map": "auto",
        "torch_dtype": "auto",
        "trust_remote_code": True,
    },
    # Memory/Embedding settings
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",  # Default embedding model
    "embedding_fallback_model": "sentence-transformers/paraphrase-MiniLM-L3-v2",  # Fallback model
}