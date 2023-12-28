from pathlib import Path
import requests
from IPython.display import clear_output, Javascript

pisah = 'https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/text_encoder/'
pisah1 = 'https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/'
donlot1 = 'https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/unet/'
donlot2 = 'https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/tokenizer/'
donlot3 = 'https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/'
donlot4 = 'https://huggingface.co/zcxu-eric/MagicAnimate/resolve/main/appearance_encoder/'
donlot5 = 'https://huggingface.co/zcxu-eric/MagicAnimate/resolve/main/densepose_controlnet/'
donlot6 = 'https://huggingface.co/zcxu-eric/MagicAnimate/resolve/main/temporal_attention/'
donlot7 = 'https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/scheduler/'

BASE_DIR = Path.cwd()

# Daftar tupel dengan nama model dan direktori tujuan
download_targets = [
    ('config.json', 'pretrained_models/stable-diffusion-v1-5/text_encoder', 'pisah'),
    ('pytorch_model.bin', 'pretrained_models/stable-diffusion-v1-5/text_encoder', 'pisah'),
    ('v1-5-pruned-emaonly.safetensors', 'pretrained_models/stable-diffusion-v1-5', 'pisah1'),

    ('config.json', 'pretrained_models/stable-diffusion-v1-5/unet', 'donlot1'),
    ('diffusion_pytorch_model.bin', 'pretrained_models/stable-diffusion-v1-5/unet', 'donlot1'),

    ('tokenizer_config.json', 'pretrained_models/stable-diffusion-v1-5/tokenizer', 'donlot2'),
    ('vocab.json', 'pretrained_models/stable-diffusion-v1-5/tokenizer', 'donlot2'),
    ('merges.txt', 'pretrained_models/stable-diffusion-v1-5/tokenizer', 'donlot2'),
    ('special_tokens_map.json', 'pretrained_models/stable-diffusion-v1-5/tokenizer', 'donlot2'),

    ('diffusion_pytorch_model.safetensors', 'pretrained_models/sd-vae-ft-mse', 'donlot3'),
    ('config.json', 'pretrained_models/sd-vae-ft-mse', 'donlot3'),

    ('diffusion_pytorch_model.safetensors', 'pretrained_models/MagicAnimate/appearance_encoder', 'donlot4'),
    ('config.json', 'pretrained_models/MagicAnimate/appearance_encoder', 'donlot4'),

    ('diffusion_pytorch_model.safetensors', 'pretrained_models/MagicAnimate/densepose_controlnet', 'donlot5'),
    ('config.json', 'pretrained_models/MagicAnimate/densepose_controlnet', 'donlot5'),

    ('temporal_attention.ckpt', 'pretrained_models/MagicAnimate/temporal_attention', 'donlot6'),

    ('scheduler_config.json', 'pretrained_models/stable-diffusion-v1-5/scheduler', 'donlot7'),
]

# Fungsi untuk mengunduh model ke direktori yang sesuai
def dl_model(link, model_name, dir_name):
    if not dir_name.exists():
        dir_name.mkdir(parents=True)

    with requests.get(f'{link}{model_name}') as r:
        r.raise_for_status()
        with open(dir_name / model_name, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

if __name__ == '__main__':
    for model, dir_path, source_url in download_targets:
        print(f'Downloading {model}...')
        dl_model(globals()[source_url], model, BASE_DIR / dir_path)
        clear_output(wait=True)

    print('All models downloaded!')
