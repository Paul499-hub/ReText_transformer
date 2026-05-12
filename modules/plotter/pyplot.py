import matplotlib.pyplot as plt
from pathlib import Path

class Plotter:
    def __init__(self):
        pass

    def maybe_loss_graph(self, 
                        step_counter:int,
                        rate:int, 
                        losses:list[float], 
                        save_path: Path,
                    ):
        if step_counter % rate == 0:
            print(f'[maybe_loss_graph] Saving plot at {save_path}')
            steps = [rate * (i + 1) for i in range(len(losses))]
            plt.figure()
            plt.plot( steps, losses)
            plt.xlabel("step")
            plt.ylabel("loss")
            plt.title("Training loss")
            plt.grid(True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
