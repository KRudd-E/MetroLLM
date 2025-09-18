
#** Parser **#
def parser():
    import argparse
    parser = argparse.ArgumentParser(description="Model Development for Text Generation")
    parser.add_argument('--run', '-r', type=str, choices=['train', 't', 'finetune', 'ft', 't', 'evaluate', 'eval', 'e'], required=True,
                        help="Specify whether to train, evaluate, or fine-tune the model.")
    parser.add_argument('--local-rank', type=int, default=-1, help='Local rank for distributed training')
    if parser.parse_args().run.lower() in ['train', 't', 'finetune', 'ft']: run = 'train'
    elif parser.parse_args().run.lower() in ['evaluate', 'eval', 'e']: run = 'evaluate'
    else: raise ValueError("Invalid run type specified. Use 'train' or 'evaluate'.")
    return run

#** Queries **#
def modelDev_textgen_query(run: str, config: dict) -> None:

    if run.lower() in ['train', 't', 'finetune', 'ft']:
        user_input = input(f"\n         ********************* >>>>> -- ModelDev TextGen -- <<<<< *********************          \n\nThis script is used to fine-tune or evaluate a Text Generation model.\nYou have opted to **Fine-tune** **{config['train']['model']['name']}**, using data at dir. **{config['train']['data']['dir']}**.\nDo you wish to continue? (y/n): ")
    if run.lower() in ['evaluate', 'eval', 'e']:
        user_input = input(f"\n         ********************* >>>>> -- ModelDev TextGen -- <<<<< *********************          \n\nThis script is used to fine-tune or evaluate a Text Generation model.\nYou have opted to **Evaluate** **{config['eval']['model']['dir']}**.\nDo you wish to continue? (y/n): ")
    if user_input.lower() in ['yes', 'y']: pass
    elif user_input.lower() in ['no', 'n']: exit()
    else: 
        while True:
            user_input = input("Invalid input. Please enter Y or N: ")
            if user_input.lower() in ['yes', 'y']:
                break
            elif user_input.lower() in ['no', 'n']:
                exit()

def training_query(self):
    while True:
        user_input = input("\nThis will fine-tune the DeepSeek-R1 Distilled model.\nBefore proceeding, ensure config.yaml contains the correct arguments.\nDo you wish proceed? (yes/no): ").strip().lower()
        if user_input.lower() in ['yes', 'y']: break
        elif user_input.lower() in ['no', 'n']: exit()
        else: print("Invalid input. Please enter yes or no.")

def evaluation_query():
    while True:
        user_input = input("\nThis will evaluate the fine-tuned DeepSeek-R1 Distilled model.\nBefore proceeding, ensure config.yaml contains the correct arguments.\nDo wish to proceed? (yes/no): ").strip().lower()
        if user_input.lower() in ['yes', 'y']: break
        elif user_input.lower() in ['no', 'n']: exit()
        else: print("Invalid input. Please enter yes or no.")


#** Config **#
def get_config(config_path='modelDev-textGen/config.yaml'):
    import yaml
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


#** Setup output dir **#
def setup_training_output_dir(self):
    """ Set up the output directory for training results.
    """
    import os
    from datetime import datetime
    import shutil
    import json
    
    self.config['train']['output_dir'] = os.path.join(self.config['train']['training_args']['output_dir'] + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '/')
    self.config['train']['log_dir'] = os.path.join(self.config['train']['output_dir']+ 'logs.ndjson')
    
    #** Make output directory **#
    os.makedirs(self.config['train']['output_dir'], exist_ok=True)
    
    #** Initialize empty log file **#
    os.makedirs(os.path.dirname(self.config['train']['log_dir']), exist_ok=True)
    with open(self.config['train']['log_dir'], 'w', encoding='utf-8') as f:
        json.dump({}, f, indent=4)

    #** Copy config file to output directory **#
    shutil.copy(self.config['config_dir'], os.path.join(self.config['train']['output_dir'] + 'config.yaml'))


#* Setup distributed processing **#
def setup_distributed():
    """Initialize distributed training"""
    import os
    import torch
    
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        
        #** Initialize **#
        torch.distributed.init_process_group(backend='nccl')
        
        #** Set device **#
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        
        print(f"Distributed training initialized. Rank: {torch.distributed.get_rank()}, "
              f"World size: {torch.distributed.get_world_size()}, "
              f"Local rank: {local_rank}")
    else:
        print("Single process training")
