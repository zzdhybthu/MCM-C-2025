from model import OlymPredictor
from dataset import OlympicDataset
from train import convert_tensor_to_serializable
from utils import *
import os
import torch
from copy import deepcopy

PT_PATH = 'ckpts/75_tensor([2000])_model.pt'
save_path = 'preds'

def predict_normal():
    os.makedirs(save_path, exist_ok=True)
    dataset = OlympicDataset(random_sample=False)
    module = OlymPredictor(device='cpu')
    module.load_params(PT_PATH)
    items_past, year = dataset[-1]
    print(year)
    
    with torch.no_grad():
        pred, noc_medal_count, Cs_history, Ss_history, Ps_history, loss, errors = module.predict2028(items_past)
    
    save_dict = {
        "noc_medal_count": noc_medal_count,
        "Cs_history": Cs_history,
        "Ss_history": Ss_history,
        "Ps_history": Ps_history,
        "pred": pred,
        "errors": errors,
        "loss": loss
    }
    print(loss)
    save_dict = convert_tensor_to_serializable(save_dict)
    write_json(save_dict, os.path.join(save_path, f"pred_{PT_PATH.replace('/', '-')[:-3]}.json"))


def predict_change():
    os.makedirs(save_path, exist_ok=True)
    dataset = OlympicDataset(random_sample=False)
    module = OlymPredictor(device='cpu')
    module.load_params(PT_PATH)
    items_past, year = dataset[-1]
    print(year)
    
    item_gd = deepcopy(items_past[-1])
    # change_sport = 'Weightlifting'
    # change_sport = 'Judo'
    change_sport = 'Shooting'
    item_gd['total_sport_medals'][change_sport] = [item_gd['total_sport_medals'][change_sport][0] * 2, item_gd['total_sport_medals'][change_sport][1] * 2, item_gd['total_sport_medals'][change_sport][2] * 2]
    
    with torch.no_grad():
        pred, noc_medal_count, Cs_history, Ss_history, Ps_history, loss, errors = module.predict2028(items_past, item_gd)
    
    save_dict = {
        "noc_medal_count": noc_medal_count,
        "Cs_history": Cs_history,
        "Ss_history": Ss_history,
        "Ps_history": Ps_history,
        "pred": pred,
        "errors": errors,
        "loss": loss
    }
    print(loss)
    save_dict = convert_tensor_to_serializable(save_dict)
    write_json(save_dict, os.path.join(save_path, f"{change_sport}_{PT_PATH.replace('/', '-')[:-3]}.json"))

if __name__ == '__main__':
    predict_normal()
    # predict_change()