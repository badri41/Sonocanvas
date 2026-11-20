// Write data/video.npz: the real waveforms and decoded images the Manim scenes need.
// The scene 3 shapes are encoded at 1.6 s, so each plays in full as it appears.
#include <format>
#include <functional>
#include <print>

#include <opencv2/imgproc.hpp>

#include "sonocanvas.hpp"

using namespace sc;

int main() {
    constexpr int W = 900, H = 260;
    constexpr double SHAPE_DUR = 1.6;
    const cv::Scalar white(255);
    const std::vector<std::pair<std::string, std::function<void(cv::Mat&)>>> shapes = {
        {"dot", [&](cv::Mat& m) { cv::ellipse(m, {450, 130}, {25, 25}, 0, 0, 360, white, cv::FILLED); }},
        {"horizontal", [&](cv::Mat& m) { cv::line(m, {120, 130}, {780, 130}, white, 7); }},
        {"vertical", [&](cv::Mat& m) { cv::line(m, {450, 30}, {450, 230}, white, 9); }},
        {"diagonal", [&](cv::Mat& m) { cv::line(m, {120, 225}, {780, 35}, white, 8); }},
        {"circle", [&](cv::Mat& m) { cv::ellipse(m, {450, 130}, {117, 102}, 0, 0, 360, white, 7); }},
    };

    Npz out("data/video.npz");
    Image img = preprocess(render_text("DA321"), "edges");
    Encoded e = encode(img);
    out.add("x", e.x);
    out.add("target", e.target);
    out.add("rec", decode(e.x));
    out.add("S", e.S);
    for (int k : {0, 1, 5, 20, 60}) out.add(std::format("gl_rec_{}", k), decode(encode(img, {}, k).x));

    Params p;
    p.dur = SHAPE_DUR;
    for (const auto& [name, draw] : shapes) {
        cv::Mat m = cv::Mat::zeros(H, W, CV_8U);
        draw(m);
        Image shape;
        m.convertTo(shape, CV_64F, 1.0 / 255);
        Encoded s = encode(shape, p);
        out.add(name + "_x", s.x);
        out.add(name + "_rec", decode(s.x, p));
    }
    std::println("wrote data/video.npz");
}
