"""The one center crop every model here is scored and run with.

Resize the short side to size / crop_frac, then cut size x size out of the
middle. At the default 0.875 (short side 183 for the 160 px students, 256
for the 224 px stock models) this is how the published numbers are
measured; crop_frac 1.0 keeps the whole frame, squared. eval_tflite.py,
distill.py's validation, classify.py and webcam.py all call this, so the
PyTorch, TFLite and deployed numbers see the same pixels.
"""
from PIL import Image

CROP_FRAC = 0.875


def center_crop(img, size, crop_frac=CROP_FRAC):
    """PIL image in, size x size PIL image out."""
    w, h = img.size
    short = int(round(size / crop_frac))
    scale = short / min(w, h)
    img = img.resize((max(size, round(w * scale)), max(size, round(h * scale))), Image.BILINEAR)
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    return img.crop((left, top, left + size, top + size))
