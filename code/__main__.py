"""

Ensure conda environment is active.
Run the program via terminal:
    user@Mac MetroLLM % code/


"""

from src.control import Controller

if __name__ == '__main__':
    controller = Controller()
    controller.run()