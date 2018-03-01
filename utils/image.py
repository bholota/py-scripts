from PIL import Image


def apply_rotation_from_original(original, target):
    if hasattr(original, '_getexif'):
        orientation = 0x0112
        exif = original._getexif()
        if exif is not None:
            orientation = exif[orientation]
            rotations = {
                3: Image.ROTATE_180,
                6: Image.ROTATE_270,
                8: Image.ROTATE_90
            }
            if orientation in rotations:
                return target.transpose(rotations[orientation])
    return target