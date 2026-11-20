from sonocanvas import decode, encode, preprocess, render_text, ssim


def test_encode_decode_da321_edges():
    img = preprocess(render_text("DA321"), mode="edges")
    x, info = encode(img)
    assert ssim(info["target"], decode(x)) > 0.9


def test_long_text_fits_canvas():
    img = render_text("DATA PROCESSING")
    assert img[:, 0].max() == 0 and img[:, -1].max() == 0
    assert img.max() == 1.0


def test_fixed_font_keeps_height_and_lengthens_clip():
    import numpy as np
    from sonocanvas import auc, text_duration

    def height(im):
        r = np.where(im.any(1))[0]
        return r[-1] - r[0]

    assert np.array_equal(render_text("HELLO", fit=False), render_text("HELLO"))  # short text: unchanged
    long = render_text("DRAWING WITH SOUND", fit=False)
    assert long.shape[1] > 900 and height(long) == height(render_text("HELLO"))
    img = preprocess(long, mode="edges")
    dur = text_duration(img)
    x, info = encode(img, dur=dur)
    assert abs(len(x) / info["sr"] - dur) < 0.05
    assert auc(info["target"], decode(x, dur=dur)) > 0.99
