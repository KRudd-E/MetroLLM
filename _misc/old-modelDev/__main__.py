from modelDev.src._control import Controller
from config import config

if __name__ == '__main__':
    controller = Controller(config)
    controller.run()