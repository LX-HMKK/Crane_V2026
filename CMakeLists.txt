cmake_minimum_required(VERSION 3.16)
project(crane_extreme CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG")

# ── options ───────────────────────────────────────────────────────────
option(CRANE_WITH_ONNXRUNTIME "Build ONNX Runtime inference backend" ON)
option(CRANE_WITH_BPU "Build RDK X5 BPU (libDNN) inference backend" ON)
option(CRANE_BUILD_TESTS "Build tests" ON)
set(ONNXRUNTIME_ROOT "${CMAKE_SOURCE_DIR}/tools/onnxruntime" CACHE PATH
    "Root of a prebuilt ONNX Runtime package (include/ + lib/)")

# Real libDNN only exists on the RDK X5 (aarch64 system image). On any other
# host we compile the BPU backend against vendored headers and link a stub so
# the full binary still builds and the backend reports itself unavailable.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64" AND EXISTS "/usr/lib/hbbpu/libdnn.so")
  set(CRANE_BPU_REAL ON)
else()
  set(CRANE_BPU_REAL OFF)
endif()
message(STATUS "CRANE_WITH_ONNXRUNTIME=${CRANE_WITH_ONNXRUNTIME}")
message(STATUS "CRANE_WITH_BPU=${CRANE_WITH_BPU} (real libDNN: ${CRANE_BPU_REAL})")

# ── dependencies ──────────────────────────────────────────────────────
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio dnn)
find_package(yaml-cpp REQUIRED)
find_package(Threads REQUIRED)

if(CRANE_WITH_ONNXRUNTIME)
  find_library(ONNXRUNTIME_LIB onnxruntime PATHS "${ONNXRUNTIME_ROOT}/lib" NO_DEFAULT_PATH)
  if(NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR
      "onnxruntime not found under ${ONNXRUNTIME_ROOT}. "
      "Run scripts/setup_onnxruntime.sh first or set -DONNXRUNTIME_ROOT=...")
  endif()
  message(STATUS "onnxruntime: ${ONNXRUNTIME_LIB}")
endif()

# ── embedded web assets ───────────────────────────────────────────────
set(WEB_ASSETS_HEADER "${CMAKE_BINARY_DIR}/generated/web_assets.hpp")
add_custom_command(
  OUTPUT "${WEB_ASSETS_HEADER}"
  COMMAND ${CMAKE_COMMAND}
          -DINPUT_HTML=${CMAKE_SOURCE_DIR}/tasks/web/index.html
          -DOUTPUT_HEADER=${WEB_ASSETS_HEADER}
          -P ${CMAKE_SOURCE_DIR}/cmake/embed_file.cmake
  DEPENDS ${CMAKE_SOURCE_DIR}/tasks/web/index.html
          ${CMAKE_SOURCE_DIR}/cmake/embed_file.cmake
  COMMENT "Embedding web assets")
add_custom_target(web_assets DEPENDS "${WEB_ASSETS_HEADER}")

# ── library (all modules live in tasks/; src/ holds only the entry) ───
set(CRANE_SOURCES
  tasks/common/config.cpp
  tasks/io/camera.cpp
  tasks/logic/mapping.cpp
  tasks/logic/logic_completion.cpp
  tasks/logic/task_generator.cpp
  tasks/io/serial_port.cpp
  tasks/inference/yolo11.cpp
  tasks/inference/model.cpp
  tasks/app.cpp
  tasks/web/http_server.cpp
  tasks/web/web_ui.cpp
)

if(CRANE_WITH_ONNXRUNTIME)
  list(APPEND CRANE_SOURCES tasks/inference/onnx_backend.cpp)
endif()

if(CRANE_WITH_BPU)
  list(APPEND CRANE_SOURCES tasks/inference/bpu_backend.cpp)
  if(NOT CRANE_BPU_REAL)
    list(APPEND CRANE_SOURCES tasks/inference/bpu_stub.c)
  endif()
endif()

add_library(crane_core STATIC ${CRANE_SOURCES})
add_dependencies(crane_core web_assets)
target_include_directories(crane_core PUBLIC
  tasks
  ${CMAKE_BINARY_DIR}/generated
  ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(crane_core PUBLIC
  ${OpenCV_LIBS}
  yaml-cpp
  Threads::Threads
)

if(CRANE_WITH_ONNXRUNTIME)
  target_compile_definitions(crane_core PUBLIC CRANE_WITH_ONNXRUNTIME=1)
  target_include_directories(crane_core PUBLIC "${ONNXRUNTIME_ROOT}/include")
  target_link_libraries(crane_core PUBLIC ${ONNXRUNTIME_LIB})
endif()

if(CRANE_WITH_BPU)
  target_compile_definitions(crane_core PUBLIC CRANE_WITH_BPU=1)
  if(CRANE_BPU_REAL)
    target_compile_definitions(crane_core PUBLIC CRANE_BPU_REAL=1)
    target_link_libraries(crane_core PUBLIC /usr/lib/hbbpu/libdnn.so)
  else()
    target_include_directories(crane_core PUBLIC tools/rdk_dnn/include)
  endif()
endif()

# ── main binary ───────────────────────────────────────────────────────
add_executable(crane_vision src/main.cpp)
target_link_libraries(crane_vision PRIVATE crane_core)
if(CRANE_WITH_ONNXRUNTIME)
  # keep the ORT shared library reachable from the binary dir
  add_custom_command(TARGET crane_vision POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_LIB}" "$<TARGET_FILE_DIR:crane_vision>")
endif()

# ── tests ─────────────────────────────────────────────────────────────
if(CRANE_BUILD_TESTS)
  enable_testing()
  add_executable(test_logic tests/test_logic.cpp)
  target_link_libraries(test_logic PRIVATE crane_core)
  add_test(NAME logic COMMAND test_logic)

  add_executable(test_serial tests/test_serial.cpp)
  target_link_libraries(test_serial PRIVATE crane_core)
  add_test(NAME serial COMMAND test_serial)

  if(CRANE_WITH_ONNXRUNTIME)
    add_executable(test_golden_onnx tests/test_golden_onnx.cpp)
    target_link_libraries(test_golden_onnx PRIVATE crane_core)
    add_test(NAME golden_onnx COMMAND test_golden_onnx
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
  endif()
endif()
