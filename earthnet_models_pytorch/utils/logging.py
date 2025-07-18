
import sys
import copy
import torch
import torchvision

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.colors as clr


CMAP_NDVI = clr.LinearSegmentedColormap.from_list('ndvi', ["#cbbe9a","#fffde4","#bccea5","#66985b","#2e6a32","#123f1e","#0e371a","#01140f","#000d0a"], N=256)
CMAP_NDVI.set_bad(color='white')

# CMAP_KNDVI = clr.LinearSegmentedColormap.from_list('kndvi', ["#cbbe9a","#fffde4","#bccea5", "#83AD75", "#689B5F", "#528A4F", "#38773B", "#286531", "#195226", "#0F441F", "#0D421D", "#0B3E1C", "#072F17", "#021D0E", "#01120B", "#010C09", "#000d0a"], N=256)
# CMAP_KNDVI.set_bad(color='white')

# CMAPS = {"ndvi": CMAP_NDVI, "kndvi": CMAP_KNDVI}

def text_phantom(text, width): #TODO move to generic function
    # Create font
    pil_font = ImageFont.load_default()#
    bbox = pil_font.getbbox(text)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]


    # create a blank canvas with extra space between lines
    canvas = Image.new('RGB', [width, text_height], (255, 255, 255))

    # draw the text onto the canvas
    draw = ImageDraw.Draw(canvas)
    offset = ((width - text_width) // 2 , 0)
    white = "#000000"
    draw.text(offset, text, font=pil_font, fill=white)

    # Convert the canvas into an array with values in [0, 1]
    return (255 - np.asarray(canvas)) / 255.0



def veg_colorize(data, mask = None, clouds = None, setting="en21x"): #TODO move to generic function or take from earthnet.tk
    # mask has 1 channel
    # clouds has 0 channel
    cmap = CMAP_NDVI # CMAPS[mode]
    t,h,w = data.shape
    in_data = copy.deepcopy(data.reshape(-1)).detach().cpu().numpy()
    if mask is not None:
        in_data = np.ma.array(in_data, mask = copy.deepcopy(mask.reshape(-1)).detach().cpu().numpy())            
    
    if clouds is None:
        return torch.as_tensor(cmap(in_data)[:,:3], dtype = data.dtype, device = data.device).reshape(t,h,w,3).permute(0,3,1,2)
    else:
        out = torch.as_tensor(cmap(in_data)[:,:3], dtype = data.dtype, device = data.device).reshape(t,h,w,3).permute(0,3,1,2)
        return torch.stack([torch.where(clouds, out[:,0,...],torch.zeros_like(out[:,0,...], dtype = data.dtype, device = data.device)), torch.where(clouds, out[:,1,...],torch.zeros_like(out[:,1,...], dtype = data.dtype, device = data.device)), torch.where(clouds, out[:,2,...],0.1*torch.ones_like(out[:,2,...], dtype = data.dtype, device = data.device))], dim = 1)



def log_viz(tensorboard_logger, viz_data, batch, batch_idx, current_epoch, setting = "en21x"):
    targs = batch["dynamic"][0]
    nrow = 9 if targs.shape[1]%9 == 0 else 10
    
    if "landcover" in batch:
        lc = batch["landcover"]
        if setting == "en21x":
            lc = 1 - ((lc >= 10).bool() & (lc <= 30).bool() ).float()
        else:
            lc = 1 - ((lc >= 2).bool() & (lc <= 6).bool()).float()  # TODO legacy have the real min_lc and max_lc
    if len(batch["dynamic_mask"]) > 0:
        if setting == "en21x":
            masks = (batch["dynamic_mask"][0] < 1).bool()
        else:
            masks = batch["dynamic_mask"][0].bool()
    else:
        masks = None
    for i, (preds, scores) in enumerate(viz_data):
        for j in range(preds.shape[0]):
            if setting == "en21x":
                ndvi = veg_colorize(preds[j,...].squeeze(), mask = lc[j,...].repeat(preds.shape[1],1,1), setting=setting)
                text = f"Cube: {scores[j]['name']} Veg Score: {scores[j]['veg_score']:.4f}" 
                grid = torchvision.utils.make_grid(ndvi, nrow = nrow)
                text = torch.tensor(text_phantom(text, width = grid.shape[-1]), dtype=torch.float32, device = targs.device).type_as(grid).permute(2,0,1)
            elif setting == "en21-std":
                rgb = torch.cat([preds[j,:,2,...].unsqueeze(1)*10000,preds[j,:,1,...].unsqueeze(1)*10000,preds[j,:,0,...].unsqueeze(1)*10000],dim = 1)
                grid = torchvision.utils.make_grid(rgb, nrow = nrow, normalize = True, value_range = (0,5000))
                text = f"Cube: {scores[j]['name']} ENS: {scores[j]['ENS']:.4f} MAD: {scores[j]['MAD']:.4f} OLS: {scores[j]['OLS']:.4f} EMD: {scores[j]['EMD']:.4f} SSIM: {scores[j]['SSIM']:.4f}"
                text = torch.tensor(text_phantom(text, width = grid.shape[-1]), dtype=torch.float32, device = targs.device).type_as(grid).permute(2,0,1)
                grid = torch.cat([grid, text], dim = -2)
                tensorboard_logger.add_image(f"Cube: {batch_idx*preds.shape[0] + j} RGB Preds, Sample: {i}", grid, current_epoch)
                ndvi = veg_colorize((preds[j,:,3,...] - preds[j,:,2,...])/(preds[j,:,3,...] + preds[j,:,2,...]+1e-6), mask = None if "landcover" not in batch else lc[j,...].repeat(preds.shape[1],1,1), setting=setting)
                grid = torchvision.utils.make_grid(ndvi, nrow = nrow)
            else:
                
                ndvi = veg_colorize(preds[j,...].squeeze(), mask = None if "landcover" not in batch else lc[j,...].repeat(preds.shape[1],1,1), setting=setting)
                text = f"Cube: {scores[j]['name']} Score: {scores[j]['rmse' if 'rmse' in scores[j] else 'veg_score']:.4f}" 
                grid = torchvision.utils.make_grid(ndvi, nrow = nrow)
                text = torch.tensor(text_phantom(text, width = grid.shape[-1]), dtype=torch.float32, device = targs.device).type_as(grid).permute(2,0,1)
            grid = torch.cat([grid, text], dim = -2)
            tensorboard_logger.add_image(f"Cube: {batch_idx*preds.shape[0] + j} NDVI Preds, Sample: {i}", grid, current_epoch)
            ndvi = (preds[j,:,3,...] - preds[j,:,2,...])/(preds[j,:,3,...] + preds[j,:,2,...]+1e-6) if setting == "en21-std" else preds[j,...].squeeze()
            ndvi_chg = (ndvi[1:,...]-ndvi[:-1,...]+1)/2  # difference between 2 consecutive predictions.
            grid = torchvision.utils.make_grid(ndvi_chg.unsqueeze(1), nrow = nrow)
            grid = torch.cat([grid, text], dim = -2)
            tensorboard_logger.add_image(f"Cube: {batch_idx*preds.shape[0] + j} NDVI Change, Sample: {i}", grid, current_epoch)
            # if j == 0:
            #     print(ndvi.shape)
            #     print("ndvi",  ndvi)
            #     print("ndvi:chg", ndvi_chg)

            # Images
            if i == 0:
                if setting == "en21x":
                    rgb = torch.cat([targs[j,:,3,...].unsqueeze(1)*10000,targs[j,:,2,...].unsqueeze(1)*10000,targs[j,:,1,...].unsqueeze(1)*10000],dim = 1)
                    grid = torchvision.utils.make_grid(rgb, nrow = nrow, normalize = True, value_range = (0,5000))
                    tensorboard_logger.add_image(f"Cube: {batch_idx*preds.shape[0] + j} RGB Targets", grid, current_epoch)

                    ndvi = veg_colorize(targs[j,:,0,...], mask = lc[j,...].repeat(targs.shape[1],1,1), clouds = masks[j,:,0,...], setting=setting)
                
                elif setting == 'en21-std':
                    rgb = torch.cat([targs[j,:,2,...].unsqueeze(1)*10000,targs[j,:,1,...].unsqueeze(1)*10000,targs[j,:,0,...].unsqueeze(1)*10000],dim = 1)
                    grid = torchvision.utils.make_grid(rgb, nrow = nrow, normalize = True, value_range = (0,5000))
                    tensorboard_logger.add_image(f"Cube: {batch_idx*preds.shape[0] + j} RGB Targets", grid, current_epoch)
                    ndvi = veg_colorize((targs[j,:,3,...] - targs[j,:,2,...])/(targs[j,:,3,...] + targs[j,:,2,...]+1e-6), mask = None if "landcover" not in batch else lc[j,...].repeat(targs.shape[1],1,1), clouds = masks[j,:,0,...] if masks is not None else None, mode = "ndvi")
                else: # kndvi      
                    ndvi = veg_colorize(targs[j,:,0,...], mask = None if "landcover" not in batch else lc[j,...].repeat(targs.shape[1],1,1), clouds = None, setting=setting)
                grid = torchvision.utils.make_grid(ndvi, nrow = nrow)
                tensorboard_logger.add_image(f"Cube: {batch_idx*preds.shape[0] + j} NDVI Targets", grid, current_epoch)
