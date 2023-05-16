from .models import opt, bloom
def from_pretrained_download(model_name, model_size, download_dir='/data/'):
    assert model_name.lower() in ['opt', 'bloom'], f'Invalid model name: {model_name}'
    if model_name.lower() == 'opt':
        opt.load_pretrained_from_size(model_size, cache_dir=download_dir)
    elif model_name.lower() == 'bloom':
        bloom.load_pretrained_from_size(model_size, cache_dir=download_dir)
