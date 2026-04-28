CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -pedantic

KERNEL_DIR := kernels
BUILD_DIR := build

.PHONY: all clean

all: $(BUILD_DIR)/pt_nonclustered $(BUILD_DIR)/pt_clustered $(BUILD_DIR)/pt_centroids_file

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/pt_nonclustered: $(KERNEL_DIR)/pt_nonclustered.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BUILD_DIR)/pt_clustered: $(KERNEL_DIR)/pt_clustered.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BUILD_DIR)/pt_centroids_file: $(KERNEL_DIR)/pt_centroids_file.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -rf $(BUILD_DIR)
