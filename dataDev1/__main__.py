from src._control import Controller
from src.utils import get_config

if __name__ == '__main__':
    config = get_config()
    controller = Controller(config)
    controller.run()