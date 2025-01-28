from model import OlymPredictor
from dataset import OlympicDataset
from train import convert_tensor_to_serializable
from utils import *
import os
import torch

PT_PATH = 'ckpts/75_tensor([2000])_model.pt'

if __name__ == '__main__':
    save_path = 'preds'
    os.makedirs(save_path, exist_ok=True)
    dataset = OlympicDataset(random_sample=False)
    module = OlymPredictor(device='cpu')
    module.load_params(PT_PATH)
    
    res = []
    for i in range(len(dataset)):
        items_past, year = dataset[i]
        print(year)
        with torch.no_grad():
            pred, noc_medal_count, Cs_history, Ss_history, Ps_history, loss, errors = module.predict2028(items_past)
        
        save_dict = {
            "noc_medal_count": noc_medal_count,
            "errors": errors,
            "loss": loss
        }
        print(loss)
        res.append(save_dict)
    res = convert_tensor_to_serializable(res)
    write_jsonl(res, os.path.join(save_path, f"pred_error_{PT_PATH.replace('/', '-')[:-3]}.jsonl"))
    
    