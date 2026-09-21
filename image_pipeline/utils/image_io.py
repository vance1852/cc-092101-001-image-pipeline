from typing import Tuple
from PIL import Image as PILImage
import os
from ..algorithms.core import Image as AlgoImage, Pixel

def pil_to_algo(pil_img: PILImage.Image) -> AlgoImage:
    if pil_img.mode not in ('RGB', 'RGBA', 'L', 'LA'):
        if 'A' in pil_img.mode:
            pil_img = pil_img.convert('RGBA')
        else:
            pil_img = pil_img.convert('RGB')
    mode = pil_img.mode
    width, height = pil_img.size
    raw = list(pil_img.getdata())
    out: AlgoImage = []
    for y in range(height):
        row = []
        for x in range(width):
            p = raw[y * width + x]
            if mode == 'L':
                row.append(int(p))
            elif mode == 'LA':
                row.append(int(p[0]))
            elif mode == 'RGB':
                row.append([int(p[0]), int(p[1]), int(p[2])])
            elif mode == 'RGBA':
                row.append([int(p[0]), int(p[1]), int(p[2]), int(p[3])])
        out.append(row)
    return out

def algo_to_pil(img: AlgoImage) -> PILImage.Image:
    if not img or not img[0]:
        return PILImage.new('L', (0, 0))
    height = len(img)
    width = len(img[0])
    sample = img[0][0]
    if isinstance(sample, int):
        mode = 'L'
        flat = []
        for row in img:
            flat.extend(row)
        pil_img = PILImage.new(mode, (width, height))
        pil_img.putdata(flat)
    else:
        channels = len(sample)
        if channels == 3:
            mode = 'RGB'
        elif channels == 4:
            mode = 'RGBA'
        else:
            mode = 'RGB'
            new_img = []
            for row in img:
                new_row = []
                for p in row:
                    if len(p) >= 3:
                        new_row.append([p[0], p[1], p[2]])
                    else:
                        g = p[0] if len(p) >= 1 else 0
                        new_row.append([g, g, g])
                new_img.append(new_row)
            img = new_img
        flat = []
        for row in img:
            for p in row:
                flat.append(tuple(p))
        pil_img = PILImage.new(mode, (width, height))
        pil_img.putdata(flat)
    return pil_img

def read_image(path: str) -> AlgoImage:
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Image file not found: {path}')
    pil_img = PILImage.open(path)
    return pil_to_algo(pil_img)

def write_image(img: AlgoImage, path: str, fmt: str=None, quality: int=90) -> None:
    pil_img = algo_to_pil(img)
    if fmt is None:
        ext = os.path.splitext(path)[1].lower()
        format_map = {'.png': 'PNG', '.jpg': 'JPEG', '.jpeg': 'JPEG', '.bmp': 'BMP', '.tif': 'TIFF', '.tiff': 'TIFF', '.webp': 'WEBP'}
        fmt = format_map.get(ext, 'PNG')
    out_dir = os.path.dirname(path)
    if out_dir and (not os.path.exists(out_dir)):
        os.makedirs(out_dir, exist_ok=True)
    if fmt == 'JPEG' and pil_img.mode in ('RGBA', 'LA', 'P'):
        bg = PILImage.new('RGB', pil_img.size, (255, 255, 255))
        if pil_img.mode == 'P':
            pil_img = pil_img.convert('RGBA')
        bg.paste(pil_img, mask=pil_img.split()[-1] if 'A' in pil_img.mode else None)
        pil_img = bg
    elif fmt == 'JPEG' and pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')
    save_kwargs = {}
    if fmt == 'JPEG':
        save_kwargs['quality'] = max(1, min(95, quality))
        save_kwargs['optimize'] = True
    elif fmt == 'PNG':
        save_kwargs['optimize'] = True
    pil_img.save(path, format=fmt, **save_kwargs)

def image_size(path: str) -> Tuple[int, int]:
    with PILImage.open(path) as img:
        return img.size

def is_valid_image(path: str) -> bool:
    try:
        with PILImage.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False
SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp', '.gif'}

def find_images(directory: str) -> list:
    if not os.path.isdir(directory):
        return []
    result = []
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            ext = os.path.splitext(name)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                result.append(path)
    return result
