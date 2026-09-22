"""Single source of truth for output filename/format resolution."""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

# Output format -> file extension. Keep in sync with the OutputNode param
# schema in nodes/definitions.py.
FORMAT_EXTENSIONS: Dict[str, str] = {
    'PNG': '.png',
    'JPEG': '.jpg',
    'BMP': '.bmp',
    'TIFF': '.tif',
    'WEBP': '.webp',
}

DEFAULT_EXTENSION = '.png'


def resolve_format(params: Dict[str, Any]) -> Optional[str]:
    """Return the normalized PIL format name, or None to keep the input ext."""
    fmt = params.get('format') if params else None
    if fmt:
        return str(fmt).upper()
    return None


def resolve_output_name(input_filename: str, params: Dict[str, Any]) -> Tuple[str, str, Optional[str]]:
    """Compute ``(output_filename, output_extension, format)`` for one node.

    Rules (identical for the OutputNode, batch preflight and dry-run):
      * an explicit ``format`` overrides the extension (PNG -> .png ...);
      * without a format the *input* extension is preserved;
      * an input without an extension defaults to .png;
      * ``suffix`` is inserted between the stem and the extension.
    """
    stem, ext = os.path.splitext(os.path.basename(input_filename))
    suffix = (params or {}).get('suffix', '') or ''
    fmt = resolve_format(params or {})
    if fmt:
        out_ext = FORMAT_EXTENSIONS.get(fmt, ext.lower() if ext else DEFAULT_EXTENSION)
    else:
        out_ext = ext.lower() if ext else DEFAULT_EXTENSION
    return f'{stem}{suffix}{out_ext}', out_ext, fmt
