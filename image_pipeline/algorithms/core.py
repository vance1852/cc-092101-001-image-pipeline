from typing import List, Tuple, Union, Callable
import math
import random
Pixel = Union[int, List[int]]
Image = List[List[Pixel]]

def _clamp(value: int, lo: int=0, hi: int=255) -> int:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value

def _is_grayscale(img: Image) -> bool:
    if not img or not img[0]:
        return True
    return isinstance(img[0][0], int)

def _image_size(img: Image) -> Tuple[int, int]:
    if not img:
        return (0, 0)
    return (len(img[0]), len(img))

def _make_image(width: int, height: int, default: Pixel=0) -> Image:
    return [[default for _ in range(width)] for _ in range(height)]

def to_grayscale(img: Image) -> Image:
    if _is_grayscale(img):
        return img
    W, H = _image_size(img)
    out = _make_image(W, H, 0)
    for y in range(H):
        for x in range(W):
            p = img[y][x]
            if len(p) >= 3:
                r, g, b = (p[0], p[1], p[2])
                L = (r * 299 + g * 587 + b * 114) // 1000
                out[y][x] = _clamp(L)
            else:
                out[y][x] = p[0] if len(p) >= 1 else 0
    return out

def adjust_brightness(img: Image, brightness: int) -> Image:
    W, H = _image_size(img)
    out = _make_image(W, H, 0 if _is_grayscale(img) else [0, 0, 0])
    if _is_grayscale(img):
        for y in range(H):
            for x in range(W):
                out[y][x] = _clamp(img[y][x] + brightness)
    else:
        channels = len(img[0][0])
        for y in range(H):
            for x in range(W):
                p = img[y][x]
                out[y][x] = [_clamp(p[c] + brightness) for c in range(channels)]
    return out

def adjust_contrast(img: Image, contrast: float) -> Image:
    W, H = _image_size(img)
    out = _make_image(W, H, 0 if _is_grayscale(img) else [0, 0, 0])
    if _is_grayscale(img):
        for y in range(H):
            for x in range(W):
                v = (img[y][x] - 128) * contrast + 128
                out[y][x] = _clamp(int(v))
    else:
        channels = len(img[0][0])
        for y in range(H):
            for x in range(W):
                p = img[y][x]
                out[y][x] = [_clamp(int((p[c] - 128) * contrast + 128)) for c in range(channels)]
    return out

def threshold(img: Image, thresh: int, mode: str='binary') -> Image:
    gray = to_grayscale(img)
    W, H = _image_size(gray)
    out = _make_image(W, H, 0)
    for y in range(H):
        for x in range(W):
            v = gray[y][x]
            if mode == 'binary':
                out[y][x] = 255 if v > thresh else 0
            elif mode == 'binary_inv':
                out[y][x] = 0 if v > thresh else 255
            elif mode == 'truncate':
                out[y][x] = thresh if v > thresh else v
            elif mode == 'tozero':
                out[y][x] = v if v > thresh else 0
            elif mode == 'tozero_inv':
                out[y][x] = 0 if v > thresh else v
            else:
                raise ValueError(f'Unknown threshold mode: {mode}')
    return out

def _normalize_kernel(kernel: List[List[float]]) -> List[List[float]]:
    s = 0.0
    for row in kernel:
        for v in row:
            s += v
    if abs(s) < 1e-09:
        return kernel
    return [[v / s for v in row] for row in kernel]

def convolve(img: Image, kernel: List[List[float]], normalize: bool=True, padding: str='reflect') -> Image:
    kh = len(kernel)
    kw = len(kernel[0]) if kh > 0 else 0
    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError(f'Kernel dimensions must be odd, got {kw}x{kh}')
    if normalize:
        K = _normalize_kernel(kernel)
    else:
        K = kernel
    ph = kh // 2
    pw = kw // 2
    W, H = _image_size(img)
    is_gray = _is_grayscale(img)
    out = _make_image(W, H, 0 if is_gray else [0, 0, 0])

    def _get_pixel(x: int, y: int) -> Pixel:
        if padding == 'reflect':
            if x < 0:
                x = -x
            elif x >= W:
                x = 2 * (W - 1) - x
            if y < 0:
                y = -y
            elif y >= H:
                y = 2 * (H - 1) - y
            return img[y][x]
        elif padding == 'zero':
            if x < 0 or x >= W or y < 0 or (y >= H):
                return 0 if is_gray else [0, 0, 0]
            return img[y][x]
        else:
            raise ValueError(f'Unknown padding: {padding}')
    if is_gray:
        for y in range(H):
            for x in range(W):
                s = 0.0
                for ky in range(kh):
                    for kx in range(kw):
                        px = _get_pixel(x + kx - pw, y + ky - ph)
                        s += px * K[ky][kx]
                out[y][x] = _clamp(int(s))
    else:
        channels = len(img[0][0])
        for y in range(H):
            for x in range(W):
                sums = [0.0] * channels
                for ky in range(kh):
                    for kx in range(kw):
                        px = _get_pixel(x + kx - pw, y + ky - ph)
                        w = K[ky][kx]
                        for c in range(channels):
                            sums[c] += px[c] * w
                out[y][x] = [_clamp(int(s)) for s in sums]
    return out

def box_blur(img: Image, size: int=3) -> Image:
    if size <= 0 or size % 2 == 0:
        raise ValueError(f'Box blur size must be positive odd integer, got {size}')
    kernel = [[1.0 for _ in range(size)] for _ in range(size)]
    return convolve(img, kernel, normalize=True)

def gaussian_blur(img: Image, size: int=3, sigma: float=1.0) -> Image:
    if size <= 0 or size % 2 == 0:
        raise ValueError(f'Gaussian kernel size must be positive odd integer, got {size}')
    if sigma <= 0:
        raise ValueError(f'Sigma must be positive, got {sigma}')
    half = size // 2
    kernel = [[0.0 for _ in range(size)] for _ in range(size)]
    two_s2 = 2.0 * sigma * sigma
    for y in range(size):
        for x in range(size):
            dx = x - half
            dy = y - half
            kernel[y][x] = math.exp(-(dx * dx + dy * dy) / two_s2)
    return convolve(img, kernel, normalize=True)

def sharpen(img: Image, amount: float=1.0) -> Image:
    if amount < 0:
        raise ValueError(f'Sharpen amount must be >= 0, got {amount}')
    c = 1.0 + 4.0 * amount
    s = -amount
    kernel = [[0.0, s, 0.0], [s, c, s], [0.0, s, 0.0]]
    return convolve(img, kernel, normalize=False)

def sobel_edges(img: Image, direction: str='both') -> Image:
    gray = to_grayscale(img)
    sx = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    sy = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
    if direction == 'x':
        return convolve(gray, sx, normalize=False, padding='reflect')
    elif direction == 'y':
        return convolve(gray, sy, normalize=False, padding='reflect')
    elif direction == 'both':
        gx = convolve(gray, sx, normalize=False, padding='reflect')
        gy = convolve(gray, sy, normalize=False, padding='reflect')
        W, H = _image_size(gray)
        out = _make_image(W, H, 0)
        for y in range(H):
            for x in range(W):
                a = gx[y][x]
                b = gy[y][x]
                m = math.sqrt(a * a + b * b)
                out[y][x] = _clamp(int(m))
        return out
    else:
        raise ValueError(f'Unknown sobel direction: {direction}')

def prewitt_edges(img: Image, direction: str='both') -> Image:
    gray = to_grayscale(img)
    sx = [[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]]
    sy = [[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
    if direction == 'x':
        return convolve(gray, sx, normalize=False)
    elif direction == 'y':
        return convolve(gray, sy, normalize=False)
    elif direction == 'both':
        gx = convolve(gray, sx, normalize=False)
        gy = convolve(gray, sy, normalize=False)
        W, H = _image_size(gray)
        out = _make_image(W, H, 0)
        for y in range(H):
            for x in range(W):
                a = gx[y][x]
                b = gy[y][x]
                m = math.sqrt(a * a + b * b)
                out[y][x] = _clamp(int(m))
        return out
    else:
        raise ValueError(f'Unknown prewitt direction: {direction}')

def crop(img: Image, x: int, y: int, width: int, height: int) -> Image:
    W, H = _image_size(img)
    if x < 0 or y < 0:
        raise ValueError(f'Crop origin must be >= 0, got ({x}, {y})')
    if width <= 0 or height <= 0:
        raise ValueError(f'Crop dimensions must be positive, got {width}x{height}')
    if x + width > W or y + height > H:
        raise ValueError(f'Crop region ({x},{y},{width},{height}) exceeds image size {W}x{H}')
    out = _make_image(width, height, 0 if _is_grayscale(img) else [0, 0, 0])
    for oy in range(height):
        for ox in range(width):
            out[oy][ox] = img[y + oy][x + ox]
    return out

def _nearest_sample(img: Image, sx: float, sy: float) -> Image:
    W, H = _image_size(img)
    new_W = max(1, int(round(W * sx)))
    new_H = max(1, int(round(H * sy)))
    is_gray = _is_grayscale(img)
    out = _make_image(new_W, new_H, 0 if is_gray else [0, 0, 0])
    for y in range(new_H):
        src_y = min(H - 1, int(y / sy))
        for x in range(new_W):
            src_x = min(W - 1, int(x / sx))
            out[y][x] = img[src_y][src_x]
    return out

def _bilinear_sample(img: Image, sx: float, sy: float) -> Image:
    W, H = _image_size(img)
    new_W = max(1, int(round(W * sx)))
    new_H = max(1, int(round(H * sy)))
    is_gray = _is_grayscale(img)
    out = _make_image(new_W, new_H, 0 if is_gray else [0, 0, 0])

    def _lerp(a, b, t):
        if is_gray:
            return int(a * (1 - t) + b * t)
        else:
            ch = len(a)
            return [int(a[c] * (1 - t) + b[c] * t) for c in range(ch)]
    for y in range(new_H):
        fy = (y + 0.5) / sy - 0.5
        y0 = int(math.floor(fy))
        y1 = y0 + 1
        ty = fy - y0
        y0 = max(0, min(H - 1, y0))
        y1 = max(0, min(H - 1, y1))
        for x in range(new_W):
            fx = (x + 0.5) / sx - 0.5
            x0 = int(math.floor(fx))
            x1 = x0 + 1
            tx = fx - x0
            x0 = max(0, min(W - 1, x0))
            x1 = max(0, min(W - 1, x1))
            a = img[y0][x0]
            b = img[y0][x1]
            c = img[y1][x0]
            d = img[y1][x1]
            top = _lerp(a, b, tx)
            bot = _lerp(c, d, tx)
            out[y][x] = _lerp(top, bot, ty)
    return out

def resize(img: Image, scale: float=None, scale_x: float=None, scale_y: float=None, target_width: int=None, target_height: int=None, method: str='bilinear') -> Image:
    W, H = _image_size(img)
    if scale is not None:
        if scale_x is None:
            scale_x = scale
        if scale_y is None:
            scale_y = scale
    if scale_x is None and scale_y is None and (target_width is None) and (target_height is None):
        raise ValueError('Must specify scale or target dimensions')
    if scale_x is not None and scale_x <= 0:
        raise ValueError(f'Scale X must be positive, got {scale_x}')
    if scale_y is not None and scale_y <= 0:
        raise ValueError(f'Scale Y must be positive, got {scale_y}')
    if target_width is not None and target_width <= 0:
        raise ValueError(f'Target width must be positive, got {target_width}')
    if target_height is not None and target_height <= 0:
        raise ValueError(f'Target height must be positive, got {target_height}')
    sx = scale_x
    sy = scale_y
    if target_width is not None:
        sx = target_width / W
    if target_height is not None:
        sy = target_height / H
    if sx is None:
        sx = sy
    if sy is None:
        sy = sx
    if method == 'nearest':
        return _nearest_sample(img, sx, sy)
    elif method == 'bilinear':
        return _bilinear_sample(img, sx, sy)
    else:
        raise ValueError(f'Unknown resize method: {method}')

def generate_gradient_image(width: int, height: int) -> Image:
    out = _make_image(width, height, [0, 0, 0])
    for y in range(height):
        for x in range(width):
            r = int(255 * x / max(1, width - 1))
            g = int(255 * y / max(1, height - 1))
            b = int(255 * (1.0 - abs((x + y) / (width + height - 2) - 0.5) * 2))
            out[y][x] = [_clamp(r), _clamp(g), _clamp(b)]
    return out

def generate_checkerboard(width: int, height: int, tile_size: int=8, color_a: List[int]=None, color_b: List[int]=None) -> Image:
    color_a = color_a or [255, 255, 255]
    color_b = color_b or [0, 0, 0]
    out = _make_image(width, height, color_a[:])
    for y in range(height):
        for x in range(width):
            tx = x // tile_size
            ty = y // tile_size
            if (tx + ty) % 2 == 1:
                out[y][x] = color_b[:]
    return out

def generate_solid_rect(width: int, height: int, rect_x: int, rect_y: int, rect_w: int, rect_h: int, bg_color: List[int]=None, rect_color: List[int]=None) -> Image:
    bg_color = bg_color or [128, 128, 128]
    rect_color = rect_color or [255, 0, 0]
    out = _make_image(width, height, bg_color[:])
    for y in range(height):
        for x in range(width):
            if rect_x <= x < rect_x + rect_w and rect_y <= y < rect_y + rect_h:
                out[y][x] = rect_color[:]
    return out

def add_noise(img: Image, amount: int=30, seed: int=42) -> Image:
    rng = random.Random(seed)
    W, H = _image_size(img)
    is_gray = _is_grayscale(img)
    out = _make_image(W, H, 0 if is_gray else [0, 0, 0])
    if is_gray:
        for y in range(H):
            for x in range(W):
                n = rng.randint(-amount, amount)
                out[y][x] = _clamp(img[y][x] + n)
    else:
        channels = len(img[0][0])
        for y in range(H):
            for x in range(W):
                p = img[y][x]
                out[y][x] = [_clamp(p[c] + rng.randint(-amount, amount)) for c in range(channels)]
    return out

def __all__() -> list:
    return ['to_grayscale', 'adjust_brightness', 'adjust_contrast', 'threshold', 'convolve', 'box_blur', 'gaussian_blur', 'sharpen', 'sobel_edges', 'prewitt_edges', 'crop', 'resize', 'generate_gradient_image', 'generate_checkerboard', 'generate_solid_rect', 'add_noise']
