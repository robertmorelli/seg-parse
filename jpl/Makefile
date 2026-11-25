CFLAGS = -O3 -std=gnu2x -Wall -Wextra -Isrc

bin/jplparse: src/main.m src/gpu.m src/cpu.c src/fuzz.c src/jpl.h src/gpu.h src/fuzz.h
	@mkdir -p bin
	clang $(CFLAGS) -fobjc-arc -o $@ src/main.m src/gpu.m src/cpu.c src/fuzz.c -framework Foundation -framework Metal

clean:
	rm -rf bin
