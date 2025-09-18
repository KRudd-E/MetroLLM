# utils/misc.py
# This script contains various utility functions used across TC-A.

def get_config(config_path:str='modelDev-textClass-A/config.yaml') -> dict:
    import yaml
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def parser() -> str:
    """ Parse command line arguments to determine run mode (train or evaluate).
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Model Development for Text Classification")
    parser.add_argument('--run', '-r', type=str, 
                        choices=['train', 't', 'finetune', 'ft', 't', 'evaluate', 'eval', 'e'], 
                        required=True,
                        help="Specify whether to train, evaluate, or fine-tune the model.")
    
    if parser.parse_args().run.lower() in ['train', 't', 'finetune', 'ft']:
        run = 'train'
    elif parser.parse_args().run.lower() in ['evaluate', 'eval', 'e']:
        run = 'evaluate'
    else:
        raise ValueError("Invalid run type specified. Use 'train' or 'evaluate'.")
    
    return run

def modelDev_textclass_query(run: str, config: dict) -> None:
    """ Query user to confirm they wish to proceed with the specified run mode and configuration.
    """
    while True:
        if run.lower() == 'train':
            user_input = input(f"\n         ********************* >>>>> -- ModelDev Text Classification A -- <<<<< *********************          \n\nThis script is used to fine-tune or evaluate a Text Classification model for the Applications database.\nYou have opted to **Fine-tune** **{config['train']['model']['name']}**, using data at dir. **{config['train']['data']['source_dir']}**.\nDo you wish to continue? (y/n): ")
        
        elif run.lower() == 'evaluate':
            user_input = input(f"\n         ********************* >>>>> -- ModelDev Text Classification -- <<<<< *********************          \n\nThis script is used to fine-tune or evaluate a Text Classification model for the Applications database.\nYou have opted to **Evaluate** **{config['eval']['model']['source_dir']}**.\nDo you wish to continue? (y/n): ")
        
        if user_input.lower() in ['yes','y']: break
        elif user_input.lower() in ['no','n']: exit()
        
        else: print("\nInvalid input. Please enter Y or N.")


def setup_training_output_dir(self) -> None:
    """ Set up the output directory for training results.
    """
    import os, shutil, json
    from datetime import datetime
    
    self.config['train']['output_dir'] = os.path.join(self.config['train']['training_args']['output_dir'] + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '/')
    self.config['train']['log_dir'] = os.path.join(self.config['train']['output_dir']+ 'logs.json')
    
    #** Make output dir **#
    os.makedirs(self.config['train']['output_dir'], exist_ok=False)
    
    #** Make empty log file **#
    os.makedirs(os.path.dirname(self.config['train']['log_dir']), exist_ok=True)
    with open(self.config['train']['log_dir'], 'w', encoding='utf-8') as f:
        json.dump({}, f, indent=4)

    #** Copy config to output dir **#
    shutil.copy(self.config['config_dir'], os.path.join(self.config['train']['output_dir'] + 'config.yaml'))
