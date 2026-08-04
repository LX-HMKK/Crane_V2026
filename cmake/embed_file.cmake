# embed_file.cmake — turn src/web/index.html into a C header with the page
# content as a raw string literal (single-binary deployment).
if(NOT INPUT_HTML OR NOT OUTPUT_HEADER)
  message(FATAL_ERROR "embed_file.cmake requires INPUT_HTML and OUTPUT_HEADER")
endif()

file(READ "${INPUT_HTML}" _html)
get_filename_component(_out_dir "${OUTPUT_HEADER}" DIRECTORY)
file(MAKE_DIRECTORY "${_out_dir}")

# A delimiter that cannot collide with page content (max 16 chars by C++).
set(_delim "CRANEHTMLx9f3b2a")
file(WRITE "${OUTPUT_HEADER}"
"// Auto-generated from src/web/index.html — do not edit.
#pragma once
namespace crane { namespace web {
inline const char* kIndexHtml = R\"${_delim}(
")
file(APPEND "${OUTPUT_HEADER}" "${_html}")
file(APPEND "${OUTPUT_HEADER}" ")${_delim}\";
}}  // namespace crane::web
")
