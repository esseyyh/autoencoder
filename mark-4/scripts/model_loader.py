

import torch
from torchsummary import summary 


from decoder import Decoder
from encoder import Encoder

import model_converter

import model_converterr2

# def preload_models_from_standard_weights(ckpt_path, device):
#     state_dict = model_converter.load_from_standard_weights(ckpt_path, device)

#     #encoder = Encoder().to(device)
#     #encoder.load_state_dict(state_dict['encoder'], strict=True)

#     decoder = Decoder().to(device)
#     decoder.load_state_dict(state_dict['decoder'], strict=True)

#     #diffusion = Diffusion().to(device)
#     #diffusion.load_state_dict(state_dict['diffusion'], strict=True)

#     #clip = CLIP().to(device)
#     #clip.load_state_dict(state_dict['clip'], strict=True)

#     return {
#         #'clip': clip,
#         #'encoder': encoder,
#         'decoder': decoder,
#         #'diffusion': diffusion,
#     }

def preload_models_from_standard_weights2(ckpt_path, device):
    state_dict = model_converterr2.load_from_standard_weights(ckpt_path, device)

    #encoder = Encoder().to(device)
    #encoder.load_state_dict(state_dict['encoder'], strict=True)

    decoder = Decoder().to(device)
    decoder.load_state_dict(state_dict['decoder'], strict=True)

    #diffusion = Diffusion().to(device)
    #diffusion.load_state_dict(state_dict['diffusion'], strict=True)

    #clip = CLIP().to(device)
    #clip.load_state_dict(state_dict['clip'], strict=True)

    return {
        #'clip': clip,
        #'encoder': encoder,
        'decoder': decoder,
        #'diffusion': diffusion,
    }


models=preload_models_from_standard_weights2("/home/essey/v1-5-pruned-emaonly.ckpt","cpu")

model=models["decoder"]
torch.save(model,"check.pt")



models=preload_models_from_standard_weights2("/home/essey/v1-5-pruned-emaonly.ckpt","cpu")

model=models["decoder"]
torch.save(model,"check.pt")





print(summary(model,(4,32,32)))