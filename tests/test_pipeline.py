from sonocanvas import decode, encode, preprocess, render_text, ssim


def test_encode_decode_da321_edges():
    img = preprocess(render_text("DA321"), mode="edges")
    x, info = encode(img)
    assert ssim(info["target"], decode(x)) > 0.9
