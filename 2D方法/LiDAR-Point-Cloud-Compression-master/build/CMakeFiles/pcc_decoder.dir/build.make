# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/cauli/Documents/2D方法平面拟合/LiDAR-Point-Cloud-Compression-master

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/cauli/Documents/2D方法平面拟合/LiDAR-Point-Cloud-Compression-master/build

# Include any dependencies generated for this target.
include CMakeFiles/pcc_decoder.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/pcc_decoder.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/pcc_decoder.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/pcc_decoder.dir/flags.make

CMakeFiles/pcc_decoder.dir/test/pcc_decoder.cpp.o: CMakeFiles/pcc_decoder.dir/flags.make
CMakeFiles/pcc_decoder.dir/test/pcc_decoder.cpp.o: ../test/pcc_decoder.cpp
CMakeFiles/pcc_decoder.dir/test/pcc_decoder.cpp.o: CMakeFiles/pcc_decoder.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/cauli/Documents/2D方法平面拟合/LiDAR-Point-Cloud-Compression-master/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/pcc_decoder.dir/test/pcc_decoder.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/pcc_decoder.dir/test/pcc_decoder.cpp.o -MF CMakeFiles/pcc_decoder.dir/test/pcc_decoder.cpp.o.d -o CMakeFiles/pcc_decoder.dir/test/pcc_decoder.cpp.o -c /home/cauli/Documents/2D方法平面拟合/LiDAR-Point-Cloud-Compression-master/test/pcc_decoder.cpp

CMakeFiles/pcc_decoder.dir/test/pcc_decoder.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pcc_decoder.dir/test/pcc_decoder.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/cauli/Documents/2D方法平面拟合/LiDAR-Point-Cloud-Compression-master/test/pcc_decoder.cpp > CMakeFiles/pcc_decoder.dir/test/pcc_decoder.cpp.i

CMakeFiles/pcc_decoder.dir/test/pcc_decoder.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pcc_decoder.dir/test/pcc_decoder.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/cauli/Documents/2D方法平面拟合/LiDAR-Point-Cloud-Compression-master/test/pcc_decoder.cpp -o CMakeFiles/pcc_decoder.dir/test/pcc_decoder.cpp.s

# Object files for target pcc_decoder
pcc_decoder_OBJECTS = \
"CMakeFiles/pcc_decoder.dir/test/pcc_decoder.cpp.o"

# External object files for target pcc_decoder
pcc_decoder_EXTERNAL_OBJECTS =

pcc_decoder: CMakeFiles/pcc_decoder.dir/test/pcc_decoder.cpp.o
pcc_decoder: CMakeFiles/pcc_decoder.dir/build.make
pcc_decoder: ../lib/libdecoder.so
pcc_decoder: /usr/lib/x86_64-linux-gnu/libopencv_shape.so.3.2.0
pcc_decoder: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.3.2.0
pcc_decoder: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.3.2.0
pcc_decoder: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.3.2.0
pcc_decoder: /usr/lib/x86_64-linux-gnu/libopencv_aruco.so.3.2.0
pcc_decoder: /usr/lib/x86_64-linux-gnu/libopencv_bgsegm.so.3.2.0
pcc_decoder: /usr/lib/x86_64-linux-gnu/libopencv_bioinspired.so.3.2.0
pcc_decoder: /usr/lib/x86_64-linux-gnu/libopencv_ccalib.so.3.2.0
pcc_decoder: /usr/lib/x86_64-linux-gnu/libopencv_datasets.so.3.2.0
pcc_decoder: /usr/lib/x86_64-linux-gnu/libopencv_dpm.so.3.2.0
pcc_decoder: /usr/lib/x86_64-linux-gnu/libopencv_face.so.3.2.0
pcc_decoder: /usr/lib/x86_64-linux-gnu/libopencv_freetype.so.3.2.0
pcc_decoder: /usr/lib/x86_64-linux-gnu/libopencv_fuzzy.so.3.2.0
pcc_decoder: /usr/lib/x86_64-linux-gnu/libopencv_hdf.so.3.2.0
pcc_decoder: /usr/lib/x86_64-linux-gnu/libopencv_line_descriptor.so.3.2.0
pcc_decoder: /usr/lib/x86_64-linux-gnu/libopencv_optflow.so.3.2.0
pcc_decoder: /usr/lib/x86_64-linux-gnu/libopencv_plot.so.3.2.0
pcc_decoder: /usr/lib/x86_64-linux-gnu/libopencv_reg.so.3.2.0
pcc_decoder: /usr/lib/x86_64-linux-gnu/libopencv_saliency.so.3.2.0
pcc_decoder: /usr/lib/x86_64-linux-gnu/libopencv_stereo.so.3.2.0
pcc_decoder: /usr/lib/x86_64-linux-gnu/libopencv_structured_light.so.3.2.0
pcc_decoder: /usr/lib/x86_64-linux-gnu/libopencv_surface_matching.so.3.2.0
pcc_decoder: /usr/lib/x86_64-linux-gnu/libopencv_text.so.3.2.0
pcc_decoder: /usr/lib/x86_64-linux-gnu/libopencv_ximgproc.so.3.2.0
pcc_decoder: /usr/lib/x86_64-linux-gnu/libopencv_xobjdetect.so.3.2.0
pcc_decoder: /usr/lib/x86_64-linux-gnu/libopencv_xphoto.so.3.2.0
pcc_decoder: /usr/lib/x86_64-linux-gnu/libboost_program_options.so
pcc_decoder: /usr/lib/x86_64-linux-gnu/libboost_system.so
pcc_decoder: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
pcc_decoder: /usr/lib/x86_64-linux-gnu/libopencv_video.so.3.2.0
pcc_decoder: /usr/lib/x86_64-linux-gnu/libopencv_viz.so.3.2.0
pcc_decoder: /usr/lib/x86_64-linux-gnu/libopencv_phase_unwrapping.so.3.2.0
pcc_decoder: /usr/lib/x86_64-linux-gnu/libopencv_rgbd.so.3.2.0
pcc_decoder: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.3.2.0
pcc_decoder: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.3.2.0
pcc_decoder: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.3.2.0
pcc_decoder: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.3.2.0
pcc_decoder: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.3.2.0
pcc_decoder: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.3.2.0
pcc_decoder: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.3.2.0
pcc_decoder: /usr/lib/x86_64-linux-gnu/libopencv_videoio.so.3.2.0
pcc_decoder: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.3.2.0
pcc_decoder: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.3.2.0
pcc_decoder: /usr/lib/x86_64-linux-gnu/libopencv_core.so.3.2.0
pcc_decoder: CMakeFiles/pcc_decoder.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/cauli/Documents/2D方法平面拟合/LiDAR-Point-Cloud-Compression-master/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable pcc_decoder"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/pcc_decoder.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/pcc_decoder.dir/build: pcc_decoder
.PHONY : CMakeFiles/pcc_decoder.dir/build

CMakeFiles/pcc_decoder.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/pcc_decoder.dir/cmake_clean.cmake
.PHONY : CMakeFiles/pcc_decoder.dir/clean

CMakeFiles/pcc_decoder.dir/depend:
	cd /home/cauli/Documents/2D方法平面拟合/LiDAR-Point-Cloud-Compression-master/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/cauli/Documents/2D方法平面拟合/LiDAR-Point-Cloud-Compression-master /home/cauli/Documents/2D方法平面拟合/LiDAR-Point-Cloud-Compression-master /home/cauli/Documents/2D方法平面拟合/LiDAR-Point-Cloud-Compression-master/build /home/cauli/Documents/2D方法平面拟合/LiDAR-Point-Cloud-Compression-master/build /home/cauli/Documents/2D方法平面拟合/LiDAR-Point-Cloud-Compression-master/build/CMakeFiles/pcc_decoder.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/pcc_decoder.dir/depend

