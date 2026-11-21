/*
SPDX-License-Identifier: MIT
Copyright (c) 2026 Badri Bishal Das

Command-line tool.

  bin/sonocanvas encode TEXT -o OUT.wav [--image FILE] [--mode edges|raw] [--fixed-font] [--png F] [--npz F]
  bin/sonocanvas decode IN.wav TEXT     [--image FILE] [--mode edges|raw] [--fixed-font] [--png F] [--npz F]

encode writes the WAV; decode scores an existing WAV against the picture TEXT (or --image) would make.
Both print AUC, SSIM and PSNR. --png writes the answer key (target above decoded); --npz writes
target, rec and the scores for other programs (the demo app reads it).
*/
#include <format>
#include <print>

#include <CLI/CLI.hpp>

#include "sonocanvas.hpp"

int main(int argc, char **argv)
{
    CLI::App app{"Hide a picture in the spectrogram of a WAV file, or score a WAV against one."};
    app.require_subcommand(1);
    std::string text, image, mode = "edges", wav, png, npz;
    bool fixed_font = false;
    for (CLI::App *sub : {app.add_subcommand("encode", "encode TEXT (or --image) into a WAV"),
                          app.add_subcommand("decode", "score the WAV IN against TEXT (or --image)")}) {
        if (sub->get_name() == "decode") sub->add_option("IN", wav, "WAV file to decode")->required();
        sub->add_option("TEXT", text, "text to draw");
        if (sub->get_name() == "encode") sub->add_option("-o,--out", wav, "output WAV")->required();
        sub->add_option("--image", image, "grayscale image instead of TEXT (always 5 s)");
        sub->add_option("--mode", mode, "preprocessing")->check(CLI::IsMember({"edges", "raw"}));
        sub->add_flag("--fixed-font", fixed_font, "keep the 170 px font for long text and lengthen the clip");
        sub->add_option("--png", png, "answer key: target above decoded");
        sub->add_option("--npz", npz, "target, rec and scores as NumPy arrays");
    }
    CLI11_PARSE(app, argc, argv);
    if (text.empty() && image.empty()) {
        std::println(stderr, "give TEXT or --image");
        return 2;
    }

    sc::Image img =
        sc::preprocess(image.empty() ? sc::render_text(text, 900, 260, !fixed_font) : sc::load(image), mode);
    double brightest;
    cv::minMaxLoc(img, nullptr, &brightest);
    if (brightest == 0) {
        std::println(stderr, "the picture has no content to encode (try the other --mode)");
        return 1;
    }
    sc::Params p;
    p.dur = image.empty() ? sc::text_duration(img) : 5.0;

    sc::Signal x;
    sc::Image target;
    double sc_err = -1;
    if (app.got_subcommand("encode")) {
        sc::Encoded e = sc::encode(img, p);
        sc::write_wav(wav, e.x, p.sr);
        x = e.x;
        target = e.target;
        sc_err = e.errs.back();
    } else {
        int sr;
        x = sc::read_wav(wav, &sr);
        if (sr != p.sr) {
            std::println(stderr, "{} is {} Hz; expected {} Hz", wav, sr, p.sr);
            return 1;
        }
        target = sc::band_image(sc::image_to_mag(img, p), p);  // the target needs no phase
    }
    sc::Image rec = sc::decode(x, p);
    double a = sc::auc(target, rec), s = sc::ssim(target, rec), q = sc::psnr(target, rec);

    std::print("{} ({:.2f} s @ {} Hz)  AUC={:.4f}  SSIM={:.4f}  PSNR={:.2f} dB", wav, x.size() / double(p.sr), p.sr, a,
               s, q);
    std::println("{}", sc_err >= 0 ? std::format("  SC={:.4f}", sc_err) : "");
    if (!png.empty())
        sc::save_figure(png, target, rec, std::format("target, 1-8 kHz over {:.1f} s", p.dur),
                        std::format("decoded from audio: AUC {:.4f}, SSIM {:.3f}", a, s));
    if (!npz.empty()) {
        sc::Npz out(npz);
        out.add("target", target);
        out.add("rec", rec);
        out.add("auc", a);
        out.add("ssim", s);
        out.add("psnr", q);
        out.add("sc", sc_err);
        out.add("dur", p.dur);
    }
}
