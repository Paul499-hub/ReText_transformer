import os
import torch
from pathlib import Path

class SaveLoader:
    def __init__(self, model, training_name:str):
        self.model = model
        self.model_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.root_dir = self._find_project_root(__file__)
        self.log_dir = self.root_dir / 'log'
        self.save_dir = self.root_dir / 'saved_models'
        self.training_name=training_name
        self.full_name=f"{self.training_name}_{self.model_param_count/1e9:.2f}B"

    def create_dirs(self):
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)
    
    def load_model_from_file(self, load_from_dir:str | Path):
        state = torch.load(load_from_dir, map_location="cpu")
        self.model.load_state_dict(state)

    def save_model(self):
        torch.save( self.model.state_dict() , self.save_dir / f"{self.full_name}.pt" )

    def append_log(self, text:str):
        log_path = self.log_dir / f"Log_{self.full_name}.txt"
        with open(log_path, 'a', encoding='utf-8') as f:
            print(text, file=f)
           

    # ------------------ HELPER / PRIVATE --------------------------------- 
    def _find_project_root(self, start: str) -> Path:
        start = Path(start).resolve().parent
        for path in [start, *start.parents]:
            if (path / "pyproject.toml").exists() or (path / "README.md").exists():
                return path
        raise FileNotFoundError("Could not find project root")