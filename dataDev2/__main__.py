from src._control import Controller
from src.utils.misc import get_config

if __name__ == '__main__':
    config = get_config()
    controller = Controller(config=config)
    controller.run()