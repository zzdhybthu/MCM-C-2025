from model import OlymPredictor
from dataset import OlympicDataset
from torch.utils.data import DataLoader
from utils import *
import torch.optim as optim
import time
import os
import torch
import random
import multiprocessing

LEARNING_RATE = 0.1
WEIGHT_DECAY = 0.0005
RANDOM_SAMPLE = True
USE_CUDA = True
PT_PATH = 'ckpts/75_tensor([2000])_model.pt'
# PT_PATH = None
EPOCH = 100

def convert_tensor_to_serializable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_tensor_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_tensor_to_serializable(item) for item in obj]
    else:
        return obj

if __name__ == '__main__':
    timestamp = time.strftime('%Y%m%d%H%M%S', time.localtime())
    log_path = os.path.join(f'log_{LEARNING_RATE}', timestamp)
    os.makedirs(log_path, exist_ok=True)
    
    seed = time.time()
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    # use_cuda = USE_CUDA and torch.cuda.is_available()
    # device = torch.device("cuda" if use_cuda else "cpu")
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = 'cpu'
    
    print('Using device', device)
    num_workers = multiprocessing.cpu_count()
    print('num workers:', num_workers)
    
    dataset = OlympicDataset(random_sample=RANDOM_SAMPLE, device=device)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    module = OlymPredictor(device)
    # module = OlymPredictor(device).to(device)
    if PT_PATH is not None:
        module.load_params(PT_PATH)
    
    print(LEARNING_RATE)
    optimizer = optim.AdamW(module.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # torch.autograd.set_detect_anomaly(True)
    
    train_iter = 0
    for epoch in range(EPOCH):
        print(f"Epoch {epoch}")
        for item, year in dataloader:
            print(year)
            module.print_params()
            pred, noc_medal_count, loss = module(item)
            save_dict = {
                "noc_medal_count": noc_medal_count,
                "gd": item[-1]['noc_medal_count'],
                "pred": pred,
                "loss": loss,
            }
            save_dict = convert_tensor_to_serializable(save_dict)
            write_json(save_dict, os.path.join(log_path, f'{train_iter}_{year}_res.json'))
            module.save_params(os.path.join(log_path, f'{train_iter}_{year}_model.pt'))
            print(f"[{train_iter}] {year} loss: {loss}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            module.init_state()
            
            train_iter += 1
    
    