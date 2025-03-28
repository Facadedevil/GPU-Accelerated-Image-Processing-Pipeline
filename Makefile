# Compiler settings
NVCC = nvcc
CXX = g++
NVCCFLAGS = -arch=sm_60 -std=c++14 -O3 -lineinfo
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra
INCLUDES = -Iinclude

# Source and object files
SRC_DIR = src
EXAMPLE_DIR = examples
TEST_DIR = tests
OBJ_DIR = obj
BIN_DIR = bin

SOURCES = $(wildcard $(SRC_DIR)/*.cu)
OBJS = $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(SOURCES))

EXAMPLES = $(wildcard $(EXAMPLE_DIR)/*.cu)
EXAMPLE_BINS = $(patsubst $(EXAMPLE_DIR)/%.cu,$(BIN_DIR)/%,$(EXAMPLES))

TESTS = $(wildcard $(TEST_DIR)/*.cu)
TEST_BINS = $(patsubst $(TEST_DIR)/%.cu,$(BIN_DIR)/%,$(TESTS))

# Output library
LIB_DIR = lib
LIB_NAME = libgpu_image_processing.a

# Create directories
$(shell mkdir -p $(BIN_DIR) $(OBJ_DIR) $(LIB_DIR))

# Default target
all: library examples tests

# Build library
library: $(LIB_DIR)/$(LIB_NAME)

$(LIB_DIR)/$(LIB_NAME): $(OBJS)
	ar rcs $@ $^

# Compile source files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

# Build examples
examples: $(EXAMPLE_BINS)

$(BIN_DIR)/%: $(EXAMPLE_DIR)/%.cu $(LIB_DIR)/$(LIB_NAME)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $< -o $@ -L$(LIB_DIR) -lgpu_image_processing

# Build tests
tests: $(TEST_BINS)

$(BIN_DIR)/%: $(TEST_DIR)/%.cu $(LIB_DIR)/$(LIB_NAME)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $< -o $@ -L$(LIB_DIR) -lgpu_image_processing

# Run all tests
run_tests: tests
	for test in $(TEST_BINS); do ./$$test; done

# Clean build files
clean:
	rm -rf $(OBJ_DIR)/* $(BIN_DIR)/* $(LIB_DIR)/*

# Clean and rebuild
rebuild: clean all

# Run benchmark examples
benchmark: examples
	$(BIN_DIR)/benchmark_example

# Show help
help:
	@echo "Available targets:"
	@echo "  all         - Build library, examples, and tests (default)"
	@echo "  library     - Build only the library"
	@echo "  examples    - Build the example programs"
	@echo "  tests       - Build the test programs"
	@echo "  run_tests   - Run all tests"
	@echo "  benchmark   - Run benchmark example"
	@echo "  clean       - Remove all build files"
	@echo "  rebuild     - Clean and rebuild everything"
	@echo "  help        - Show this help message"

.PHONY: all library examples tests run_tests clean rebuild benchmark help