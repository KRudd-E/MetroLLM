from src.control import Controller
from config_ import config

if __name__ == '__main__':
    config = config
    controller = Controller(config=config)
    print(f'\nRunning dataDev\n')
    controller.run()