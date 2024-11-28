import os
def set_env(level=2):
    cache=f"{'../'*level}hf_cache"
    os.environ['HF_HOME']=cache
    os.environ['HF_DATASETS_CACHE']=cache
