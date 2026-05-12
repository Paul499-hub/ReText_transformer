import matplotlib.pyplot as plt

class Plotter:
    def __init__(self):
        pass

    def show_loss_graph(self, list_float:list[float], step_multiplier:int ):
        steps = [step_multiplier * (i + 1) for i in range(len(list_float))]
        plt.plot( steps, list_float)
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.title("Training loss")
        plt.grid(True)
        plt.show()