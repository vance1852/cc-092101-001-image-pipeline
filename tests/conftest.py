import os
import sys
import json
import tempfile
import shutil
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from image_pipeline.algorithms import core as alg
from image_pipeline.utils.image_io import write_image
from image_pipeline.utils.sample_generator import generate_all

@pytest.fixture
def tmpdir_path(tmp_path):
    return str(tmp_path)

@pytest.fixture
def sample_data_dir(tmpdir_path):
    info = generate_all(tmpdir_path)
    return info

@pytest.fixture
def simple_pipeline_config():
    return {'version': '1.0', 'name': 'test_pipeline', 'nodes': [{'id': 'in', 'type': 'input'}, {'id': 'gray', 'type': 'grayscale'}, {'id': 'thresh', 'type': 'threshold', 'params': {'value': 128}}, {'id': 'blur', 'type': 'box_blur', 'params': {'size': 3}}, {'id': 'out', 'type': 'output', 'params': {'suffix': '_out'}}], 'edges': [{'from': 'in', 'to': 'gray'}, {'from': 'gray', 'to': 'thresh'}, {'from': 'thresh', 'to': 'blur'}, {'from': 'blur', 'to': 'out'}]}

@pytest.fixture
def checkerboard_img():
    return alg.generate_checkerboard(16, 16, tile_size=4)

@pytest.fixture
def gradient_img():
    return alg.generate_gradient_image(8, 8)

def _write_corrupt_image(path):
    with open(path, 'wb') as f:
        f.write(b'This is not a valid PNG image at all! ' * 20)

@pytest.fixture
def mixed_input_dir(tmpdir_path):
    in_dir = os.path.join(tmpdir_path, 'input_mixed')
    os.makedirs(in_dir, exist_ok=True)
    img1 = alg.generate_gradient_image(8, 8)
    write_image(img1, os.path.join(in_dir, 'good1.png'), fmt='PNG')
    _write_corrupt_image(os.path.join(in_dir, 'bad.png'))
    img2 = alg.generate_checkerboard(12, 12, tile_size=3)
    write_image(img2, os.path.join(in_dir, 'good2.png'), fmt='PNG')
    return in_dir

@pytest.fixture
def all_corrupt_dir(tmpdir_path):
    in_dir = os.path.join(tmpdir_path, 'input_all_bad')
    os.makedirs(in_dir, exist_ok=True)
    _write_corrupt_image(os.path.join(in_dir, 'bad1.png'))
    _write_corrupt_image(os.path.join(in_dir, 'bad2.png'))
    return in_dir

def _write_bom_config(path, config_dict):
    content = json.dumps(config_dict, indent=2)
    bom_bytes = b'\xef\xbb\xbf'
    with open(path, 'wb') as f:
        f.write(bom_bytes)
        f.write(content.encode('utf-8'))

@pytest.fixture
def bom_config_path(tmpdir_path, simple_pipeline_config):
    p = os.path.join(tmpdir_path, 'pipeline_bom.json')
    _write_bom_config(p, simple_pipeline_config)
    return p
