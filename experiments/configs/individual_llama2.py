import os
#所有的文件的路径都是以调用这个文件的源文件的路径作为初始路径,也就是说被调用的文件的路径相对的是源文件的路径
# os.sys.path.append("..")
# # print(os.getcwd())
# print(os.sys.path)
# print(os.path.abspath('..'))
from configs.template import get_config as default_config

def get_config():

    config = default_config()
    config.result_prefix = '../results/individual_llama2'
    config.tokenizer_paths=["../../llama-2/llama/llama-2-7b-chat-hf"]
    config.model_paths=["../../llama-2/llama/llama-2-7b-chat-hf"]
    config.conversation_templates=['llama-2']

    return config