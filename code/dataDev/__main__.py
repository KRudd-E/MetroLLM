from src.control import Controller
from config import config

if __name__ == '__main__':
    config = config
    controller = Controller(config=config)
    controller.run()