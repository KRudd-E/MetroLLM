def parser():
    import argparse
    parser = argparse.ArgumentParser(description="Model Development for Text Classication")
    parser.add_argument('--run', '-r', type=str, choices=['train', 't', 'finetune', 'ft', 't', 'evaluate', 'eval', 'e'], required=True,
                        help="Specify whether to train, evaluate, or fine-tune the model.")
    if parser.parse_args().run.lower() in ['train', 't', 'finetune', 'ft']:
        run = 'train'
    elif parser.parse_args().run.lower() in ['evaluate', 'eval', 'e']:
        run = 'evaluate'
    else:
        raise ValueError("Invalid run type specified. Use 'train' or 'evaluate'.")
    return run

def modelDev_textclass_query(run: str, config: dict) -> None:
    while True:
        if run.lower() == 'train':
            user_input = input(f"\n         ********************* >>>>> -- ModelDev Text Classification -- <<<<< *********************          \n\nThis script is used to fine-tune or evaluate a Text Classification model.\nYou have opted to **Fine-tune** **{config['train']['model']['name']}**, using data at dir. **{config['train']['data']['dir']}**.\nDo you wish to continue? (y/n): ")
        if run.lower() == 'evaluate':
            user_input = input(f"\n         ********************* >>>>> -- ModelDev Text Classification -- <<<<< *********************          \n\nThis script is used to fine-tune or evaluate a Text Classification model.\nYou have opted to **Evaluate** **{config['eval']['model']['dir']}**.\nDo you wish to continue? (y/n): ")
        if user_input.lower() in ['yes','y']: break
        elif user_input.lower() in ['no','n']: exit()
        else: print("\nInvalid input. Please enter Y or N.")

def get_config(config_path='modelDev-textClass/config.yaml'):
    import yaml
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config
