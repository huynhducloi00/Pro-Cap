from utils import set_env


set_env(4)
from roberta_dataset import Multimodal_Data
from utils import set_seed

from llm_dataset import LLM_Data
import pbm
import torch

import config
from train import train_for_epoch
from torch.utils.data import DataLoader

if __name__=='__main__':
    opt=config.parse_opt()
    torch.cuda.set_device(opt.CUDA_DEVICE)
    set_seed(opt.SEED)
    
    data_class=LLM_Data if opt.LLM else Multimodal_Data
    # Create tokenizer 
    constructor='build_baseline'
    if opt.MODEL=='pbm':
        train_set=data_class(opt,'train')
        test_set=data_class(opt,'test')
        
        max_length=opt.LENGTH+opt.CAP_LENGTH
        #for one example, default 50
        #default, meme text, caption plus template
        """
        basically, length for one example: meme_text and caption
        """
        if opt.ASK_CAP!='':
            num_ask_cap=len(opt.ASK_CAP.split(','))
            all_cap_len=opt.CAP_LENGTH * num_ask_cap #default, 12*5=60
            max_length+=all_cap_len
        if opt.NUM_MEME_CAP>0:
            max_length+=opt.NUM_MEME_CAP*opt.CAP_LENGTH #default, 12*x
        if opt.USE_DEMO:
            max_length*=(opt.NUM_SAMPLE*opt.NUM_LABELS+1)
            
        label_words=[opt.POS_WORD,opt.NEG_WORD]
        model=getattr(pbm,constructor)(label_words,max_length).cuda()
    train_set=[train_set.convert_to_item(i) for i in range(len(train_set.entries))]
    test_set=[test_set.convert_to_item(i) for i in range(len(test_set.entries))]
    torch.save(train_set, f'{opt.DATASET}_trainset_llm.pkl');torch.save(test_set, f'{opt.DATASET}_testset_llm.pkl');print('xong');exit(0)
    
    train_loader=DataLoader(train_set,
                            opt.BATCH_SIZE,
                            shuffle=True)
    test_loader=DataLoader(test_set,
                           opt.BATCH_SIZE,
                           shuffle=False)
    
    train_for_epoch(opt,model,train_loader,test_loader)
    
    exit(0)
    