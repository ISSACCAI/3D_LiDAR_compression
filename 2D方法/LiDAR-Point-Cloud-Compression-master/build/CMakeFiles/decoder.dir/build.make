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
include CMakeFiles/decoder.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/decoder.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/decoder.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/decoder.dir/flags.make

CMakeFiles/decoder.dir/src/decoder.cpp.o: CMakeFiles/decoder.dir/flags.make
CMakeFiles/decoder.dir/src/decoder.cpp.o: ../src/decoder.cpp
CMakeFiles/decoder.dir/src/decoder.cpp.o: CMakeFiles/decoder.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/cauli/Documents/2D方法平面拟合/LiDAR-Point-Cloud-Compression-master/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/decoder.dir/src/decoder.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/decoder.dir/src/decoder.cpp.o -MF CMakeFiles/decoder.dir/src/decoder.cpp.o.d -o CMakeFiles/decoder.dir/src/decoder.cpp.o -c /home/cauli/Documents/2D方法平面拟合/LiDAR-Point-Cloud-Compression-master/src/decoder.cpp

CMakeFiles/decoder.dir/src/decoder.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/decoder.dir/src/decoder.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/cauli/Documents/2D方法平面拟合/LiDAR-Point-Cloud-Compression-master/src/decoder.cpp > CMakeFiles/decoder.dir/src/decoder.cpp.i

CMakeFiles/decoder.dir/src/decoder.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/decoder.dir/src/decoder.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/cauli/Documents/2D方法平面拟合/LiDAR-Point-Cloud-Compression-master/src/decoder.cpp -o CMakeFiles/decoder.dir/src/decoder.cpp.s

CMakeFiles/decoder.dir/src/encoder.cpp.o: CMakeFiles/decoder.dir/flags.make
CMakeFiles/decoder.dir/src/encoder.cpp.o: ../src/encoder.cpp
CMakeFiles/decoder.dir/src/encoder.cpp.o: CMakeFiles/decoder.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/cauli/Documents/2D方法平面拟合/LiDAR-Point-Cloud-Compression-master/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/decoder.dir/src/encoder.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/decoder.dir/src/encoder.cpp.o -MF CMakeFiles/decoder.dir/src/encoder.cpp.o.d -o CMakeFiles/decoder.dir/src/encoder.cpp.o -c /home/cauli/Documents/2D方法平面拟合/LiDAR-Point-Cloud-Compression-master/src/encoder.cpp

CMakeFiles/decoder.dir/src/encoder.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/decoder.dir/src/encoder.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/cauli/Documents/2D方法平面拟合/LiDAR-Point-Cloud-Compression-master/src/encoder.cpp > CMakeFiles/decoder.dir/src/encoder.cpp.i

CMakeFiles/decoder.dir/src/encoder.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/decoder.dir/src/encoder.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/cauli/Documents/2D方法平面拟合/LiDAR-Point-Cloud-Compression-master/src/encoder.cpp -o CMakeFiles/decoder.dir/src/encoder.cpp.s

CMakeFiles/decoder.dir/src/io.cpp.o: CMakeFiles/decoder.dir/flags.make
CMakeFiles/decoder.dir/src/io.cpp.o: ../src/io.cpp
CMakeFiles/decoder.dir/src/io.cpp.o: CMakeFiles/decoder.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/cauli/Documents/2D方法平面拟合/LiDAR-Point-Cloud-Compression-master/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/decoder.dir/src/io.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/decoder.dir/src/io.cpp.o -MF CMakeFiles/decoder.dir/src/io.cpp.o.d -o CMakeFiles/decoder.dir/src/io.cpp.o -c /home/cauli/Documents/2D方法平面拟合/LiDAR-Point-Cloud-Compression-master/src/io.cpp

CMakeFiles/decoder.dir/src/io.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/decoder.dir/src/io.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/cauli/Documents/2D方法平面拟合/LiDAR-Point-Cloud-Compression-master/src/io.cpp > CMakeFiles/decoder.dir/src/io.cpp.i

CMakeFiles/decoder.dir/src/io.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/decoder.dir/src/io.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/cauli/Documents/2D方法平面拟合/LiDAR-Point-Cloud-Compression-master/src/io.cpp -o CMakeFiles/decoder.dir/src/io.cpp.s

CMakeFiles/decoder.dir/src/pcc_module.cpp.o: CMakeFiles/decoder.dir/flags.make
CMakeFiles/decoder.dir/src/pcc_module.cpp.o: ../src/pcc_module.cpp
CMakeFiles/decoder.dir/src/pcc_module.cpp.o: CMakeFiles/decoder.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/cauli/Documents/2D方法平面拟合/LiDAR-Point-Cloud-Compression-master/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/decoder.dir/src/pcc_module.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/decoder.dir/src/pcc_module.cpp.o -MF CMakeFiles/decoder.dir/src/pcc_module.cpp.o.d -o CMakeFiles/decoder.dir/src/pcc_module.cpp.o -c /home/cauli/Documents/2D方法平面拟合/LiDAR-Point-Cloud-Compression-master/src/pcc_module.cpp

CMakeFiles/decoder.dir/src/pcc_module.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/decoder.dir/src/pcc_module.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/cauli/Documents/2D方法平面拟合/LiDAR-Point-Cloud-Compression-master/src/pcc_module.cpp > CMakeFiles/decoder.dir/src/pcc_module.cpp.i

CMakeFiles/decoder.dir/src/pcc_module.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/decoder.dir/src/pcc_module.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/cauli/Documents/2D方法平面拟合/LiDAR-Point-Cloud-Compression-master/src/pcc_module.cpp -o CMakeFiles/decoder.dir/src/pcc_module.cpp.s

CMakeFiles/decoder.dir/src/utils.cpp.o: CMakeFiles/decoder.dir/flags.make
CMakeFiles/decoder.dir/src/utils.cpp.o: ../src/utils.cpp
CMakeFiles/decoder.dir/src/utils.cpp.o: CMakeFiles/decoder.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/cauli/Documents/2D方法平面拟合/LiDAR-Point-Cloud-Compression-master/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/decoder.dir/src/utils.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/decoder.dir/src/utils.cpp.o -MF CMakeFiles/decoder.dir/src/utils.cpp.o.d -o CMakeFiles/decoder.dir/src/utils.cpp.o -c /home/cauli/Documents/2D方法平面拟合/LiDAR-Point-Cloud-Compression-master/src/utils.cpp

CMakeFiles/decoder.dir/src/utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/decoder.dir/src/utils.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/cauli/Documents/2D方法平面拟合/LiDAR-Point-Cloud-Compression-master/src/utils.cpp > CMakeFiles/decoder.dir/src/utils.cpp.i

CMakeFiles/decoder.dir/src/utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/decoder.dir/src/utils.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/cauli/Documents/2D方法平面拟合/LiDAR-Point-Cloud-Compression-master/src/utils.cpp -o CMakeFiles/decoder.dir/src/utils.cpp.s

# Object files for target decoder
decoder_OBJECTS = \
"CMakeFiles/decoder.dir/src/decoder.cpp.o" \
"CMakeFiles/decoder.dir/src/encoder.cpp.o" \
"CMakeFiles/decoder.dir/src/io.cpp.o" \
"CMakeFiles/decoder.dir/src/pcc_module.cpp.o" \
"CMakeFiles/decoder.dir/src/utils.cpp.o"

# External object files for target decoder
decoder_EXTERNAL_OBJECTS =

../lib/libdecoder.so: CMakeFiles/decoder.dir/src/decoder.cpp.o
../lib/libdecoder.so: CMakeFiles/decoder.dir/src/encoder.cpp.o
../lib/libdecoder.so: CMakeFiles/decoder.dir/src/io.cpp.o
../lib/libdecoder.so: CMakeFiles/decoder.dir/src/pcc_module.cpp.o
../lib/libdecoder.so: CMakeFiles/decoder.dir/src/utils.cpp.o
../lib/libdecoder.so: CMakeFiles/decoder.dir/build.make
../lib/libdecoder.so: /usr/lib/x86_64-linux-gnu/libopencv_shape.so.3.2.0
../lib/libdecoder.so: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.3.2.0
../lib/libdecoder.so: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.3.2.0
../lib/libdecoder.so: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.3.2.0
../lib/libdecoder.so: /usr/lib/x86_64-linux-gnu/libopencv_aruco.so.3.2.0
../lib/libdecoder.so: /usr/lib/x86_64-linux-gnu/libopencv_bgsegm.so.3.2.0
../lib/libdecoder.so: /usr/lib/x86_64-linux-gnu/libopencv_bioinspired.so.3.2.0
../lib/libdecoder.so: /usr/lib/x86_64-linux-gnu/libopencv_ccalib.so.3.2.0
../lib/libdecoder.so: /usr/lib/x86_64-linux-gnu/libopencv_datasets.so.3.2.0
../lib/libdecoder.so: /usr/lib/x86_64-linux-gnu/libopencv_dpm.so.3.2.0
../lib/libdecoder.so: /usr/lib/x86_64-linux-gnu/libopencv_face.so.3.2.0
../lib/libdecoder.so: /usr/lib/x86_64-linux-gnu/libopencv_freetype.so.3.2.0
../lib/libdecoder.so: /usr/lib/x86_64-linux-gnu/libopencv_fuzzy.so.3.2.0
../lib/libdecoder.so: /usr/lib/x86_64-linux-gnu/libopencv_hdf.so.3.2.0
../lib/libdecoder.so: /usr/lib/x86_64-linux-gnu/libopencv_line_descriptor.so.3.2.0
../lib/libdecoder.so: /usr/lib/x86_64-linux-gnu/libopencv_optflow.so.3.2.0
../lib/libdecoder.so: /usr/lib/x86_64-linux-gnu/libopencv_plot.so.3.2.0
../lib/libdecoder.so: /usr/lib/x86_64-linux-gnu/libopencv_reg.so.3.2.0
../lib/libdecoder.so: /usr/lib/x86_64-linux-gnu/libopencv_saliency.so.3.2.0
../lib/libdecoder.so: /usr/lib/x86_64-linux-gnu/libopencv_stereo.so.3.2.0
../lib/libdecoder.so: /usr/lib/x86_64-linux-gnu/libopencv_structured_light.so.3.2.0
../lib/libdecoder.so: /usr/lib/x86_64-linux-gnu/libopencv_surface_matching.so.3.2.0
../lib/libdecoder.so: /usr/lib/x86_64-linux-gnu/libopencv_text.so.3.2.0
../lib/libdecoder.so: /usr/lib/x86_64-linux-gnu/libopencv_ximgproc.so.3.2.0
../lib/libdecoder.so: /usr/lib/x86_64-linux-gnu/libopencv_xobjdetect.so.3.2.0
../lib/libdecoder.so: /usr/lib/x86_64-linux-gnu/libopencv_xphoto.so.3.2.0
../lib/libdecoder.so: /usr/lib/x86_64-linux-gnu/libboost_program_options.so
../lib/libdecoder.so: /usr/lib/x86_64-linux-gnu/libboost_system.so
../lib/libdecoder.so: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
../lib/libdecoder.so: /usr/lib/x86_64-linux-gnu/libopencv_video.so.3.2.0
../lib/libdecoder.so: /usr/lib/x86_64-linux-gnu/libopencv_viz.so.3.2.0
../lib/libdecoder.so: /usr/lib/x86_64-linux-gnu/libopencv_phase_unwrapping.so.3.2.0
../lib/libdecoder.so: /usr/lib/x86_64-linux-gnu/libopencv_rgbd.so.3.2.0
../lib/libdecoder.so: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.3.2.0
../lib/libdecoder.so: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.3.2.0
../lib/libdecoder.so: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.3.2.0
../lib/libdecoder.so: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.3.2.0
../lib/libdecoder.so: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.3.2.0
../lib/libdecoder.so: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.3.2.0
../lib/libdecoder.so: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.3.2.0
../lib/libdecoder.so: /usr/lib/x86_64-linux-gnu/libopencv_videoio.so.3.2.0
../lib/libdecoder.so: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.3.2.0
../lib/libdecoder.so: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.3.2.0
../lib/libdecoder.so: /usr/lib/x86_64-linux-gnu/libopencv_core.so.3.2.0
../lib/libdecoder.so: CMakeFiles/decoder.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/cauli/Documents/2D方法平面拟合/LiDAR-Point-Cloud-Compression-master/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Linking CXX shared library ../lib/libdecoder.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/decoder.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/decoder.dir/build: ../lib/libdecoder.so
.PHONY : CMakeFiles/decoder.dir/build

CMakeFiles/decoder.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/decoder.dir/cmake_clean.cmake
.PHONY : CMakeFiles/decoder.dir/clean

CMakeFiles/decoder.dir/depend:
	cd /home/cauli/Documents/2D方法平面拟合/LiDAR-Point-Cloud-Compression-master/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/cauli/Documents/2D方法平面拟合/LiDAR-Point-Cloud-Compression-master /home/cauli/Documents/2D方法平面拟合/LiDAR-Point-Cloud-Compression-master /home/cauli/Documents/2D方法平面拟合/LiDAR-Point-Cloud-Compression-master/build /home/cauli/Documents/2D方法平面拟合/LiDAR-Point-Cloud-Compression-master/build /home/cauli/Documents/2D方法平面拟合/LiDAR-Point-Cloud-Compression-master/build/CMakeFiles/decoder.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/decoder.dir/depend

