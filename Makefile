CXX      := clang++
CXXFLAGS := -std=c++23 -O2 -Wall -Wextra -Isrc -I/opt/homebrew/include/opencv5 -I/opt/homebrew/include \
            -DSONOCANVAS_FONT=\"$(CURDIR)/src/fonts/DejaVuSans-Bold.ttf\"
LDLIBS   := -L/opt/homebrew/lib -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_freetype \
            -lopencv_geometry -lsndfile -lz
PY       := .venv/bin/python

LIB  := src/sonocanvas.cpp src/third_party/cnpy.cpp
OBJ  := $(LIB:src/%.cpp=build/%.o)
BINS := bin/sonocanvas bin/experiments bin/analyze_recording bin/make_numbers bin/make_assets bin/test_core

.PHONY: all help test clean
all: $(BINS)

help:
	@echo "make              build bin/*"
	@echo "make test         test the app"
	@echo "make clean        remove the compiled binaries"

build/%.o: src/%.cpp src/sonocanvas.hpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

bin/%: build/%.o $(OBJ)
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDLIBS)

# cnpy is third-party code for numpy support in C++

build/third_party/cnpy.o: CXXFLAGS += -w
build/main.o build/test_core.o: CXXFLAGS += -Wno-unused-parameter
bin/sonocanvas: build/main.o $(OBJ)
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDLIBS)

test: bin/test_core
	bin/test_core
	$(PY) -m pytest -q tests

clean:
	rm -rf build bin
