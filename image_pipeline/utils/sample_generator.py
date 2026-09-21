from typing import Dict, Any, List
import os
import json
from ..algorithms import core as alg
from ..utils.image_io import write_image
from ..config.loader import sample_pipeline_config, sample_complex_pipeline_config, cycle_pipeline_config, invalid_params_pipeline_config

def generate_sample_images(output_dir: str) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    generated = []
    img_gradient = alg.generate_gradient_image(8, 8)
    p1 = os.path.join(output_dir, 'grad_8x8.png')
    write_image(img_gradient, p1, fmt='PNG')
    generated.append(p1)
    img_checker = alg.generate_checkerboard(16, 16, tile_size=8)
    p2 = os.path.join(output_dir, 'checker_16x16.png')
    write_image(img_checker, p2, fmt='PNG')
    generated.append(p2)
    img_rect = alg.generate_solid_rect(width=12, height=12, rect_x=3, rect_y=2, rect_w=6, rect_h=7, bg_color=[128, 128, 128], rect_color=[255, 0, 0])
    p3 = os.path.join(output_dir, 'rect_12x12.png')
    write_image(img_rect, p3, fmt='PNG')
    generated.append(p3)
    img_noisy = alg.add_noise(alg.generate_gradient_image(16, 16), amount=40, seed=123)
    p4 = os.path.join(output_dir, 'noisy_grad_16x16.png')
    write_image(img_noisy, p4, fmt='PNG')
    generated.append(p4)
    tiny = [[[0, 0, 0], [64, 64, 64], [128, 128, 128], [192, 192, 192], [255, 255, 255]], [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255]], [[10, 20, 30], [40, 50, 60], [70, 80, 90], [100, 110, 120], [130, 140, 150]], [[200, 100, 50], [150, 200, 100], [100, 150, 200], [50, 100, 150], [25, 75, 125]], [[255, 255, 255], [0, 0, 0], [255, 255, 255], [0, 0, 0], [255, 255, 255]]]
    p5 = os.path.join(output_dir, 'tiny_5x5.png')
    write_image(tiny, p5, fmt='PNG')
    generated.append(p5)
    img_large = alg.generate_gradient_image(32, 32)
    p6 = os.path.join(output_dir, 'grad_32x32.png')
    write_image(img_large, p6, fmt='PNG')
    generated.append(p6)
    return generated

def generate_sample_configs(output_dir: str) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    generated = []
    p1 = os.path.join(output_dir, 'pipeline_simple.json')
    with open(p1, 'w', encoding='utf-8') as f:
        json.dump(sample_pipeline_config(), f, indent=2)
    generated.append(p1)
    p2 = os.path.join(output_dir, 'pipeline_complex.json')
    with open(p2, 'w', encoding='utf-8') as f:
        json.dump(sample_complex_pipeline_config(), f, indent=2)
    generated.append(p2)
    p3 = os.path.join(output_dir, 'pipeline_cyclic_INVALID.json')
    with open(p3, 'w', encoding='utf-8') as f:
        json.dump(cycle_pipeline_config(), f, indent=2)
    generated.append(p3)
    p4 = os.path.join(output_dir, 'pipeline_bad_params_INVALID.json')
    with open(p4, 'w', encoding='utf-8') as f:
        json.dump(invalid_params_pipeline_config(), f, indent=2)
    generated.append(p4)
    return generated

def generate_all(base_dir: str, images_subdir: str='input_images', configs_subdir: str='pipelines') -> Dict[str, Any]:
    images_dir = os.path.join(base_dir, images_subdir)
    configs_dir = os.path.join(base_dir, configs_subdir)
    images = generate_sample_images(images_dir)
    configs = generate_sample_configs(configs_dir)
    output_dir = os.path.join(base_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    return {'base_dir': os.path.abspath(base_dir), 'input_dir': os.path.abspath(images_dir), 'output_dir': os.path.abspath(output_dir), 'pipeline_dir': os.path.abspath(configs_dir), 'images': images, 'configs': configs, 'image_count': len(images), 'config_count': len(configs)}
