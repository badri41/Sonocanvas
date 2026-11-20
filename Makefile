# SonoCanvas build and pipeline. `make` builds the C++ programs; `make help` lists the targets.

CXX      ?= clang++
CXXFLAGS := -std=c++23 -O2 -Wall -Wextra -Icpp -I/opt/homebrew/include/opencv5 -I/opt/homebrew/include
LDLIBS   := -L/opt/homebrew/lib -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_freetype \
            -lopencv_geometry -lsndfile -lz
PY       := .venv/bin/python

LIB  := cpp/stft.cpp cpp/phase.cpp cpp/freq_map.cpp cpp/image_prep.cpp cpp/metrics.cpp \
        cpp/sonocanvas.cpp cpp/io.cpp cpp/third_party/cnpy.cpp
OBJ  := $(LIB:cpp/%.cpp=build/%.o)
BINS := bin/sonocanvas bin/experiments bin/analyze_recording bin/make_numbers bin/make_assets bin/test_core

.PHONY: all help test experiments figures numbers paper assets video demo final clean
all: $(BINS)

help:
	@echo "make              build bin/*"
	@echo "make test         C++ unit tests, then the app tests"
	@echo "make experiments  run all six experiments (data/*.npz, results/table.csv), then figures"
	@echo "make figures      redraw figures/ from data/*.npz"
	@echo "make numbers      paper macros, Table I and the README results table"
	@echo "make paper        numbers, then paper/main.pdf"
	@echo "make assets       data/video.npz for the Manim scenes"
	@echo "make video        render the 1080p video (video/render.sh h)"
	@echo "make demo         regenerate demo/*.wav"
	@echo "make final        assemble the presentation kit in final/"

build/%.o: cpp/%.cpp cpp/sonocanvas.hpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

bin/%: build/%.o $(OBJ)
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDLIBS)

# Our own warnings only: cnpy is third-party code.
build/third_party/cnpy.o: CXXFLAGS += -w
build/main.o build/test_core.o: CXXFLAGS += -Wno-unused-parameter
bin/sonocanvas: build/main.o $(OBJ)
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDLIBS)

test: bin/test_core
	bin/test_core
	$(PY) -m pytest -q tests

experiments: bin/experiments
	bin/experiments
	$(MAKE) figures

figures:
	$(PY) plots.py

numbers: bin/make_numbers
	bin/make_numbers

paper: numbers
	cd paper && tectonic main.tex

assets: bin/make_assets
	bin/make_assets

video: assets
	video/render.sh h

demo: bin/sonocanvas
	for w in DA321 HELLO IITG; do bin/sonocanvas encode $$w -o demo/$$w.wav; done

final: bin/sonocanvas
	rm -rf final && mkdir -p final/wav
	cp paper/main.pdf final/SonoCanvas_paper.pdf
	cp DEMO_CHECKLIST.md final/
	for w in DA321 HELLO IITG MMDP GABOR PIXEL; do \
	    bin/sonocanvas encode $$w -o final/wav/$$w.wav --png final/wav/$$w.png; done
	for w in DA321 HELLO IITG; do cmp final/wav/$$w.wav demo/$$w.wav; done  # the app's fallback files
	if [ -f video/sonocanvas_final_1080p.mp4 ]; then cp video/sonocanvas_final_1080p.mp4 final/; \
	else echo "No voiced video yet: copying the silent draft."; \
	     cp video/sonocanvas_1080p60.mp4 final/sonocanvas_DRAFT_no_voice_1080p.mp4; fi
	du -sh final

clean:
	rm -rf build bin
