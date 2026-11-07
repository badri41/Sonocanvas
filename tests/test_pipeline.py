from sonocanvas import decode, encode, preprocess, render_text, ssim


def test_encode_decode_da321_edges():
    img = preprocess(render_text("DA321"), mode="edges")
    x, info = encode(img)
    assert ssim(info["target"], decode(x)) > 0.9


def test_long_text_fits_canvas():
    img = render_text("DATA PROCESSING")
    assert img[:, 0].max() == 0 and img[:, -1].max() == 0
    assert img.max() == 1.0
