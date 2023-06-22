# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/nvidia/Documents/jetson_multimedia_api/argus

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/nvidia/Documents/jetson_multimedia_api/argus

# Include any dependencies generated for this target.
include samples/rawBayerOutput/CMakeFiles/argus_rawBayerOutput.dir/depend.make

# Include the progress variables for this target.
include samples/rawBayerOutput/CMakeFiles/argus_rawBayerOutput.dir/progress.make

# Include the compile flags for this target's objects.
include samples/rawBayerOutput/CMakeFiles/argus_rawBayerOutput.dir/flags.make

samples/rawBayerOutput/CMakeFiles/argus_rawBayerOutput.dir/main.cpp.o: samples/rawBayerOutput/CMakeFiles/argus_rawBayerOutput.dir/flags.make
samples/rawBayerOutput/CMakeFiles/argus_rawBayerOutput.dir/main.cpp.o: samples/rawBayerOutput/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nvidia/Documents/jetson_multimedia_api/argus/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object samples/rawBayerOutput/CMakeFiles/argus_rawBayerOutput.dir/main.cpp.o"
	cd /home/nvidia/Documents/jetson_multimedia_api/argus/samples/rawBayerOutput && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/argus_rawBayerOutput.dir/main.cpp.o -c /home/nvidia/Documents/jetson_multimedia_api/argus/samples/rawBayerOutput/main.cpp

samples/rawBayerOutput/CMakeFiles/argus_rawBayerOutput.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/argus_rawBayerOutput.dir/main.cpp.i"
	cd /home/nvidia/Documents/jetson_multimedia_api/argus/samples/rawBayerOutput && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nvidia/Documents/jetson_multimedia_api/argus/samples/rawBayerOutput/main.cpp > CMakeFiles/argus_rawBayerOutput.dir/main.cpp.i

samples/rawBayerOutput/CMakeFiles/argus_rawBayerOutput.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/argus_rawBayerOutput.dir/main.cpp.s"
	cd /home/nvidia/Documents/jetson_multimedia_api/argus/samples/rawBayerOutput && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nvidia/Documents/jetson_multimedia_api/argus/samples/rawBayerOutput/main.cpp -o CMakeFiles/argus_rawBayerOutput.dir/main.cpp.s

# Object files for target argus_rawBayerOutput
argus_rawBayerOutput_OBJECTS = \
"CMakeFiles/argus_rawBayerOutput.dir/main.cpp.o"

# External object files for target argus_rawBayerOutput
argus_rawBayerOutput_EXTERNAL_OBJECTS =

samples/rawBayerOutput/argus_rawBayerOutput: samples/rawBayerOutput/CMakeFiles/argus_rawBayerOutput.dir/main.cpp.o
samples/rawBayerOutput/argus_rawBayerOutput: samples/rawBayerOutput/CMakeFiles/argus_rawBayerOutput.dir/build.make
samples/rawBayerOutput/argus_rawBayerOutput: /usr/lib/aarch64-linux-gnu/tegra/libnvargus_socketclient.so
samples/rawBayerOutput/argus_rawBayerOutput: /usr/lib/aarch64-linux-gnu/libopencv_gapi.so.4.5.4
samples/rawBayerOutput/argus_rawBayerOutput: /usr/lib/aarch64-linux-gnu/libopencv_highgui.so.4.5.4
samples/rawBayerOutput/argus_rawBayerOutput: /usr/lib/aarch64-linux-gnu/libopencv_ml.so.4.5.4
samples/rawBayerOutput/argus_rawBayerOutput: /usr/lib/aarch64-linux-gnu/libopencv_objdetect.so.4.5.4
samples/rawBayerOutput/argus_rawBayerOutput: /usr/lib/aarch64-linux-gnu/libopencv_photo.so.4.5.4
samples/rawBayerOutput/argus_rawBayerOutput: /usr/lib/aarch64-linux-gnu/libopencv_stitching.so.4.5.4
samples/rawBayerOutput/argus_rawBayerOutput: /usr/lib/aarch64-linux-gnu/libopencv_video.so.4.5.4
samples/rawBayerOutput/argus_rawBayerOutput: /usr/lib/aarch64-linux-gnu/libopencv_videoio.so.4.5.4
samples/rawBayerOutput/argus_rawBayerOutput: /usr/lib/aarch64-linux-gnu/libGLESv2.so.2
samples/rawBayerOutput/argus_rawBayerOutput: samples/utils/libargussampleutils.a
samples/rawBayerOutput/argus_rawBayerOutput: /usr/lib/aarch64-linux-gnu/libopencv_imgcodecs.so.4.5.4
samples/rawBayerOutput/argus_rawBayerOutput: /usr/lib/aarch64-linux-gnu/libopencv_dnn.so.4.5.4
samples/rawBayerOutput/argus_rawBayerOutput: /usr/lib/aarch64-linux-gnu/libopencv_calib3d.so.4.5.4
samples/rawBayerOutput/argus_rawBayerOutput: /usr/lib/aarch64-linux-gnu/libopencv_features2d.so.4.5.4
samples/rawBayerOutput/argus_rawBayerOutput: /usr/lib/aarch64-linux-gnu/libopencv_flann.so.4.5.4
samples/rawBayerOutput/argus_rawBayerOutput: /usr/lib/aarch64-linux-gnu/libopencv_imgproc.so.4.5.4
samples/rawBayerOutput/argus_rawBayerOutput: /usr/lib/aarch64-linux-gnu/libopencv_core.so.4.5.4
samples/rawBayerOutput/argus_rawBayerOutput: /usr/local/cuda-11.4/lib64/libcudart_static.a
samples/rawBayerOutput/argus_rawBayerOutput: /usr/lib/aarch64-linux-gnu/librt.so
samples/rawBayerOutput/argus_rawBayerOutput: /usr/local/cuda-11.4/lib64/libcudart_static.a
samples/rawBayerOutput/argus_rawBayerOutput: /usr/lib/aarch64-linux-gnu/librt.so
samples/rawBayerOutput/argus_rawBayerOutput: /usr/lib/aarch64-linux-gnu/libSM.so
samples/rawBayerOutput/argus_rawBayerOutput: /usr/lib/aarch64-linux-gnu/libICE.so
samples/rawBayerOutput/argus_rawBayerOutput: /usr/lib/aarch64-linux-gnu/libX11.so
samples/rawBayerOutput/argus_rawBayerOutput: /usr/lib/aarch64-linux-gnu/libXext.so
samples/rawBayerOutput/argus_rawBayerOutput: /usr/lib/aarch64-linux-gnu/libEGL.so
samples/rawBayerOutput/argus_rawBayerOutput: /usr/lib/aarch64-linux-gnu/libGLESv2.so.2
samples/rawBayerOutput/argus_rawBayerOutput: /usr/lib/aarch64-linux-gnu/tegra/libnvbufsurface.so
samples/rawBayerOutput/argus_rawBayerOutput: samples/rawBayerOutput/CMakeFiles/argus_rawBayerOutput.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/nvidia/Documents/jetson_multimedia_api/argus/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable argus_rawBayerOutput"
	cd /home/nvidia/Documents/jetson_multimedia_api/argus/samples/rawBayerOutput && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/argus_rawBayerOutput.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
samples/rawBayerOutput/CMakeFiles/argus_rawBayerOutput.dir/build: samples/rawBayerOutput/argus_rawBayerOutput

.PHONY : samples/rawBayerOutput/CMakeFiles/argus_rawBayerOutput.dir/build

samples/rawBayerOutput/CMakeFiles/argus_rawBayerOutput.dir/clean:
	cd /home/nvidia/Documents/jetson_multimedia_api/argus/samples/rawBayerOutput && $(CMAKE_COMMAND) -P CMakeFiles/argus_rawBayerOutput.dir/cmake_clean.cmake
.PHONY : samples/rawBayerOutput/CMakeFiles/argus_rawBayerOutput.dir/clean

samples/rawBayerOutput/CMakeFiles/argus_rawBayerOutput.dir/depend:
	cd /home/nvidia/Documents/jetson_multimedia_api/argus && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/nvidia/Documents/jetson_multimedia_api/argus /home/nvidia/Documents/jetson_multimedia_api/argus/samples/rawBayerOutput /home/nvidia/Documents/jetson_multimedia_api/argus /home/nvidia/Documents/jetson_multimedia_api/argus/samples/rawBayerOutput /home/nvidia/Documents/jetson_multimedia_api/argus/samples/rawBayerOutput/CMakeFiles/argus_rawBayerOutput.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : samples/rawBayerOutput/CMakeFiles/argus_rawBayerOutput.dir/depend
