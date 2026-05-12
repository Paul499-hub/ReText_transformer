
import torch
import time
# Modules
from modules.utils.SaveLoader import SaveLoader
from modules.encoders.MathEncoder import MathEncoder
class Monitor:
    def __init__(self):
        pass

    def maybe_log(  self, 
                    save_loader:SaveLoader, 
                    enc:MathEncoder,
                    logits: torch.Tensor,
                    loss:torch.Tensor,
                    loss_avg:list,
                    losses:list,
                    q_0_str:str,
                    a_0_str:str,
                    s_t:float,
                    tokens_seen_window:int,
                    step_counter:int, 
                    stop_step:int, 
                    rate:int, 
                ):
        # Show info, save plots
        if step_counter % rate == 0:
            time_elapsed_window = time.time() - s_t
            s_t = time.time()
            tokens_seen_save = tokens_seen_window
            tokens_seen_window = 0
            loss_avg.append(sum(losses) / len(losses))
            losses.clear()
            probs = torch.softmax(logits, dim=-1)
            pred_ids = torch.argmax(probs, dim=-1)
            text = enc.decode(pred_ids[0].tolist()) # <-- batch aware text decode
            progress = (step_counter / stop_step) * 100
            # Print
            log_text = '\n'.join([
                '\n',
                "=" * 60,
                f"🦵 steps {step_counter}/{stop_step} | {progress:.2f}%",
                f"⏱️ elapsed_window: {time_elapsed_window:2f}s",
                f'⌚ tok/s: {tokens_seen_save/ time_elapsed_window:.2f}',
                f"❓ Q : {q_0_str}",
                f" ---> 🎯 A : {a_0_str}",
                f" ---> 🦑 P : {text}",
                f"📉 loss: {loss.item():.4f}",
                f"📊 avg : {loss_avg[-1]:.4f}",
                "=" * 60
            ])
            print(f'[progress] step {step_counter}/{stop_step} | {progress:.2f}%')
            save_loader.append_log(log_text)
        return s_t, tokens_seen_window

    def maybe_save(self, 
                   save_loader:SaveLoader,
                   step_counter:int, 
                   rate:int, 
                ):
        if step_counter % rate == 0:
            print(f'[maybe_save] Saving current model in saved_models...')
            save_loader.save_model()