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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/luis/cmake/EVM

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/luis/cmake/EVM/build

# Include any dependencies generated for this target.
include CMakeFiles/evm-cpp-linux.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/evm-cpp-linux.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/evm-cpp-linux.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/evm-cpp-linux.dir/flags.make

CMakeFiles/evm-cpp-linux.dir/src/cmplx.c.o: CMakeFiles/evm-cpp-linux.dir/flags.make
CMakeFiles/evm-cpp-linux.dir/src/cmplx.c.o: ../src/cmplx.c
CMakeFiles/evm-cpp-linux.dir/src/cmplx.c.o: CMakeFiles/evm-cpp-linux.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/luis/cmake/EVM/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/evm-cpp-linux.dir/src/cmplx.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/evm-cpp-linux.dir/src/cmplx.c.o -MF CMakeFiles/evm-cpp-linux.dir/src/cmplx.c.o.d -o CMakeFiles/evm-cpp-linux.dir/src/cmplx.c.o -c /home/luis/cmake/EVM/src/cmplx.c

CMakeFiles/evm-cpp-linux.dir/src/cmplx.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/evm-cpp-linux.dir/src/cmplx.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/luis/cmake/EVM/src/cmplx.c > CMakeFiles/evm-cpp-linux.dir/src/cmplx.c.i

CMakeFiles/evm-cpp-linux.dir/src/cmplx.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/evm-cpp-linux.dir/src/cmplx.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/luis/cmake/EVM/src/cmplx.c -o CMakeFiles/evm-cpp-linux.dir/src/cmplx.c.s

CMakeFiles/evm-cpp-linux.dir/src/const.c.o: CMakeFiles/evm-cpp-linux.dir/flags.make
CMakeFiles/evm-cpp-linux.dir/src/const.c.o: ../src/const.c
CMakeFiles/evm-cpp-linux.dir/src/const.c.o: CMakeFiles/evm-cpp-linux.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/luis/cmake/EVM/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object CMakeFiles/evm-cpp-linux.dir/src/const.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/evm-cpp-linux.dir/src/const.c.o -MF CMakeFiles/evm-cpp-linux.dir/src/const.c.o.d -o CMakeFiles/evm-cpp-linux.dir/src/const.c.o -c /home/luis/cmake/EVM/src/const.c

CMakeFiles/evm-cpp-linux.dir/src/const.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/evm-cpp-linux.dir/src/const.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/luis/cmake/EVM/src/const.c > CMakeFiles/evm-cpp-linux.dir/src/const.c.i

CMakeFiles/evm-cpp-linux.dir/src/const.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/evm-cpp-linux.dir/src/const.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/luis/cmake/EVM/src/const.c -o CMakeFiles/evm-cpp-linux.dir/src/const.c.s

CMakeFiles/evm-cpp-linux.dir/src/ellf.c.o: CMakeFiles/evm-cpp-linux.dir/flags.make
CMakeFiles/evm-cpp-linux.dir/src/ellf.c.o: ../src/ellf.c
CMakeFiles/evm-cpp-linux.dir/src/ellf.c.o: CMakeFiles/evm-cpp-linux.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/luis/cmake/EVM/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building C object CMakeFiles/evm-cpp-linux.dir/src/ellf.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/evm-cpp-linux.dir/src/ellf.c.o -MF CMakeFiles/evm-cpp-linux.dir/src/ellf.c.o.d -o CMakeFiles/evm-cpp-linux.dir/src/ellf.c.o -c /home/luis/cmake/EVM/src/ellf.c

CMakeFiles/evm-cpp-linux.dir/src/ellf.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/evm-cpp-linux.dir/src/ellf.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/luis/cmake/EVM/src/ellf.c > CMakeFiles/evm-cpp-linux.dir/src/ellf.c.i

CMakeFiles/evm-cpp-linux.dir/src/ellf.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/evm-cpp-linux.dir/src/ellf.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/luis/cmake/EVM/src/ellf.c -o CMakeFiles/evm-cpp-linux.dir/src/ellf.c.s

CMakeFiles/evm-cpp-linux.dir/src/im_conv.cpp.o: CMakeFiles/evm-cpp-linux.dir/flags.make
CMakeFiles/evm-cpp-linux.dir/src/im_conv.cpp.o: ../src/im_conv.cpp
CMakeFiles/evm-cpp-linux.dir/src/im_conv.cpp.o: CMakeFiles/evm-cpp-linux.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/luis/cmake/EVM/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/evm-cpp-linux.dir/src/im_conv.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/evm-cpp-linux.dir/src/im_conv.cpp.o -MF CMakeFiles/evm-cpp-linux.dir/src/im_conv.cpp.o.d -o CMakeFiles/evm-cpp-linux.dir/src/im_conv.cpp.o -c /home/luis/cmake/EVM/src/im_conv.cpp

CMakeFiles/evm-cpp-linux.dir/src/im_conv.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/evm-cpp-linux.dir/src/im_conv.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/luis/cmake/EVM/src/im_conv.cpp > CMakeFiles/evm-cpp-linux.dir/src/im_conv.cpp.i

CMakeFiles/evm-cpp-linux.dir/src/im_conv.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/evm-cpp-linux.dir/src/im_conv.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/luis/cmake/EVM/src/im_conv.cpp -o CMakeFiles/evm-cpp-linux.dir/src/im_conv.cpp.s

CMakeFiles/evm-cpp-linux.dir/src/main.cpp.o: CMakeFiles/evm-cpp-linux.dir/flags.make
CMakeFiles/evm-cpp-linux.dir/src/main.cpp.o: ../src/main.cpp
CMakeFiles/evm-cpp-linux.dir/src/main.cpp.o: CMakeFiles/evm-cpp-linux.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/luis/cmake/EVM/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/evm-cpp-linux.dir/src/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/evm-cpp-linux.dir/src/main.cpp.o -MF CMakeFiles/evm-cpp-linux.dir/src/main.cpp.o.d -o CMakeFiles/evm-cpp-linux.dir/src/main.cpp.o -c /home/luis/cmake/EVM/src/main.cpp

CMakeFiles/evm-cpp-linux.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/evm-cpp-linux.dir/src/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/luis/cmake/EVM/src/main.cpp > CMakeFiles/evm-cpp-linux.dir/src/main.cpp.i

CMakeFiles/evm-cpp-linux.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/evm-cpp-linux.dir/src/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/luis/cmake/EVM/src/main.cpp -o CMakeFiles/evm-cpp-linux.dir/src/main.cpp.s

CMakeFiles/evm-cpp-linux.dir/src/mtherr.c.o: CMakeFiles/evm-cpp-linux.dir/flags.make
CMakeFiles/evm-cpp-linux.dir/src/mtherr.c.o: ../src/mtherr.c
CMakeFiles/evm-cpp-linux.dir/src/mtherr.c.o: CMakeFiles/evm-cpp-linux.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/luis/cmake/EVM/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building C object CMakeFiles/evm-cpp-linux.dir/src/mtherr.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/evm-cpp-linux.dir/src/mtherr.c.o -MF CMakeFiles/evm-cpp-linux.dir/src/mtherr.c.o.d -o CMakeFiles/evm-cpp-linux.dir/src/mtherr.c.o -c /home/luis/cmake/EVM/src/mtherr.c

CMakeFiles/evm-cpp-linux.dir/src/mtherr.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/evm-cpp-linux.dir/src/mtherr.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/luis/cmake/EVM/src/mtherr.c > CMakeFiles/evm-cpp-linux.dir/src/mtherr.c.i

CMakeFiles/evm-cpp-linux.dir/src/mtherr.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/evm-cpp-linux.dir/src/mtherr.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/luis/cmake/EVM/src/mtherr.c -o CMakeFiles/evm-cpp-linux.dir/src/mtherr.c.s

CMakeFiles/evm-cpp-linux.dir/src/processing_functions.cpp.o: CMakeFiles/evm-cpp-linux.dir/flags.make
CMakeFiles/evm-cpp-linux.dir/src/processing_functions.cpp.o: ../src/processing_functions.cpp
CMakeFiles/evm-cpp-linux.dir/src/processing_functions.cpp.o: CMakeFiles/evm-cpp-linux.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/luis/cmake/EVM/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/evm-cpp-linux.dir/src/processing_functions.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/evm-cpp-linux.dir/src/processing_functions.cpp.o -MF CMakeFiles/evm-cpp-linux.dir/src/processing_functions.cpp.o.d -o CMakeFiles/evm-cpp-linux.dir/src/processing_functions.cpp.o -c /home/luis/cmake/EVM/src/processing_functions.cpp

CMakeFiles/evm-cpp-linux.dir/src/processing_functions.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/evm-cpp-linux.dir/src/processing_functions.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/luis/cmake/EVM/src/processing_functions.cpp > CMakeFiles/evm-cpp-linux.dir/src/processing_functions.cpp.i

CMakeFiles/evm-cpp-linux.dir/src/processing_functions.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/evm-cpp-linux.dir/src/processing_functions.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/luis/cmake/EVM/src/processing_functions.cpp -o CMakeFiles/evm-cpp-linux.dir/src/processing_functions.cpp.s

# Object files for target evm-cpp-linux
evm__cpp__linux_OBJECTS = \
"CMakeFiles/evm-cpp-linux.dir/src/cmplx.c.o" \
"CMakeFiles/evm-cpp-linux.dir/src/const.c.o" \
"CMakeFiles/evm-cpp-linux.dir/src/ellf.c.o" \
"CMakeFiles/evm-cpp-linux.dir/src/im_conv.cpp.o" \
"CMakeFiles/evm-cpp-linux.dir/src/main.cpp.o" \
"CMakeFiles/evm-cpp-linux.dir/src/mtherr.c.o" \
"CMakeFiles/evm-cpp-linux.dir/src/processing_functions.cpp.o"

# External object files for target evm-cpp-linux
evm__cpp__linux_EXTERNAL_OBJECTS =

evm-cpp-linux: CMakeFiles/evm-cpp-linux.dir/src/cmplx.c.o
evm-cpp-linux: CMakeFiles/evm-cpp-linux.dir/src/const.c.o
evm-cpp-linux: CMakeFiles/evm-cpp-linux.dir/src/ellf.c.o
evm-cpp-linux: CMakeFiles/evm-cpp-linux.dir/src/im_conv.cpp.o
evm-cpp-linux: CMakeFiles/evm-cpp-linux.dir/src/main.cpp.o
evm-cpp-linux: CMakeFiles/evm-cpp-linux.dir/src/mtherr.c.o
evm-cpp-linux: CMakeFiles/evm-cpp-linux.dir/src/processing_functions.cpp.o
evm-cpp-linux: CMakeFiles/evm-cpp-linux.dir/build.make
evm-cpp-linux: /usr/local/lib/libopencv_gapi.so.4.7.0
evm-cpp-linux: /usr/local/lib/libopencv_stitching.so.4.7.0
evm-cpp-linux: /usr/local/lib/libopencv_aruco.so.4.7.0
evm-cpp-linux: /usr/local/lib/libopencv_barcode.so.4.7.0
evm-cpp-linux: /usr/local/lib/libopencv_bgsegm.so.4.7.0
evm-cpp-linux: /usr/local/lib/libopencv_bioinspired.so.4.7.0
evm-cpp-linux: /usr/local/lib/libopencv_ccalib.so.4.7.0
evm-cpp-linux: /usr/local/lib/libopencv_dnn_objdetect.so.4.7.0
evm-cpp-linux: /usr/local/lib/libopencv_dnn_superres.so.4.7.0
evm-cpp-linux: /usr/local/lib/libopencv_dpm.so.4.7.0
evm-cpp-linux: /usr/local/lib/libopencv_face.so.4.7.0
evm-cpp-linux: /usr/local/lib/libopencv_freetype.so.4.7.0
evm-cpp-linux: /usr/local/lib/libopencv_fuzzy.so.4.7.0
evm-cpp-linux: /usr/local/lib/libopencv_hfs.so.4.7.0
evm-cpp-linux: /usr/local/lib/libopencv_img_hash.so.4.7.0
evm-cpp-linux: /usr/local/lib/libopencv_intensity_transform.so.4.7.0
evm-cpp-linux: /usr/local/lib/libopencv_line_descriptor.so.4.7.0
evm-cpp-linux: /usr/local/lib/libopencv_mcc.so.4.7.0
evm-cpp-linux: /usr/local/lib/libopencv_quality.so.4.7.0
evm-cpp-linux: /usr/local/lib/libopencv_rapid.so.4.7.0
evm-cpp-linux: /usr/local/lib/libopencv_reg.so.4.7.0
evm-cpp-linux: /usr/local/lib/libopencv_rgbd.so.4.7.0
evm-cpp-linux: /usr/local/lib/libopencv_saliency.so.4.7.0
evm-cpp-linux: /usr/local/lib/libopencv_stereo.so.4.7.0
evm-cpp-linux: /usr/local/lib/libopencv_structured_light.so.4.7.0
evm-cpp-linux: /usr/local/lib/libopencv_superres.so.4.7.0
evm-cpp-linux: /usr/local/lib/libopencv_surface_matching.so.4.7.0
evm-cpp-linux: /usr/local/lib/libopencv_tracking.so.4.7.0
evm-cpp-linux: /usr/local/lib/libopencv_videostab.so.4.7.0
evm-cpp-linux: /usr/local/lib/libopencv_wechat_qrcode.so.4.7.0
evm-cpp-linux: /usr/local/lib/libopencv_xfeatures2d.so.4.7.0
evm-cpp-linux: /usr/local/lib/libopencv_xobjdetect.so.4.7.0
evm-cpp-linux: /usr/local/lib/libopencv_xphoto.so.4.7.0
evm-cpp-linux: /usr/local/lib/libopencv_shape.so.4.7.0
evm-cpp-linux: /usr/local/lib/libopencv_highgui.so.4.7.0
evm-cpp-linux: /usr/local/lib/libopencv_datasets.so.4.7.0
evm-cpp-linux: /usr/local/lib/libopencv_plot.so.4.7.0
evm-cpp-linux: /usr/local/lib/libopencv_text.so.4.7.0
evm-cpp-linux: /usr/local/lib/libopencv_ml.so.4.7.0
evm-cpp-linux: /usr/local/lib/libopencv_phase_unwrapping.so.4.7.0
evm-cpp-linux: /usr/local/lib/libopencv_optflow.so.4.7.0
evm-cpp-linux: /usr/local/lib/libopencv_ximgproc.so.4.7.0
evm-cpp-linux: /usr/local/lib/libopencv_video.so.4.7.0
evm-cpp-linux: /usr/local/lib/libopencv_videoio.so.4.7.0
evm-cpp-linux: /usr/local/lib/libopencv_imgcodecs.so.4.7.0
evm-cpp-linux: /usr/local/lib/libopencv_objdetect.so.4.7.0
evm-cpp-linux: /usr/local/lib/libopencv_calib3d.so.4.7.0
evm-cpp-linux: /usr/local/lib/libopencv_dnn.so.4.7.0
evm-cpp-linux: /usr/local/lib/libopencv_features2d.so.4.7.0
evm-cpp-linux: /usr/local/lib/libopencv_flann.so.4.7.0
evm-cpp-linux: /usr/local/lib/libopencv_photo.so.4.7.0
evm-cpp-linux: /usr/local/lib/libopencv_imgproc.so.4.7.0
evm-cpp-linux: /usr/local/lib/libopencv_core.so.4.7.0
evm-cpp-linux: /usr/lib/gcc/x86_64-linux-gnu/11/libgomp.so
evm-cpp-linux: /usr/lib/x86_64-linux-gnu/libpthread.a
evm-cpp-linux: CMakeFiles/evm-cpp-linux.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/luis/cmake/EVM/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Linking CXX executable evm-cpp-linux"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/evm-cpp-linux.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/evm-cpp-linux.dir/build: evm-cpp-linux
.PHONY : CMakeFiles/evm-cpp-linux.dir/build

CMakeFiles/evm-cpp-linux.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/evm-cpp-linux.dir/cmake_clean.cmake
.PHONY : CMakeFiles/evm-cpp-linux.dir/clean

CMakeFiles/evm-cpp-linux.dir/depend:
	cd /home/luis/cmake/EVM/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/luis/cmake/EVM /home/luis/cmake/EVM /home/luis/cmake/EVM/build /home/luis/cmake/EVM/build /home/luis/cmake/EVM/build/CMakeFiles/evm-cpp-linux.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/evm-cpp-linux.dir/depend

