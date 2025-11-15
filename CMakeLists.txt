cmake_minimum_required(VERSION 3.15)
project(ExynosX940CompleteV3 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
option(ENABLE_SHADER_BUILD "Compile shaders at build time (requires glslc)" ON)

include_directories(include)

file(GLOB_RECURSE SRC src/*.c src/*.h)
add_library(exynosx940complete SHARED ${SRC})

if(ENABLE_SHADER_BUILD)
    find_program(GLSLC glslc)
    find_program(XXD xxd)
    if(GLSLC AND XXD)
        file(GLOB SHADERS assets/shaders/*.comp)
        foreach(SH ${SHADERS})
            get_filename_component(NAME ${SH} NAME_WE)
            set(SPV ${CMAKE_CURRENT_BINARY_DIR}/${NAME}.spv)
            add_custom_command(OUTPUT ${SPV}
                COMMAND ${GLSLC} -fshader-stage=compute ${CMAKE_CURRENT_SOURCE_DIR}/${SH} -o ${SPV}
                DEPENDS ${SH}
            )
            set(HEADER ${CMAKE_CURRENT_BINARY_DIR}/${NAME}.spv.h)
            add_custom_command(OUTPUT ${HEADER}
                COMMAND ${XXD} -i ${SPV} > ${HEADER}
                DEPENDS ${SPV}
            )
            list(APPEND SPV_HEADERS ${HEADER})
        endforeach()
        add_custom_target(compile_shaders DEPENDS ${SPV_HEADERS})
        add_dependencies(exynosx940complete compile_shaders)
        include_directories(${CMAKE_CURRENT_BINARY_DIR})
    else()
        message(WARNING "glslc or xxd not found; runtime SPIR-V files will be used instead.")
    endif()
endif()

find_path(VULKAN_INCLUDE vulkan/vulkan.h)
find_library(VULKAN_LIB NAMES vulkan vk)
if(VULKAN_INCLUDE) target_include_directories(exynosx940complete PRIVATE ${VULKAN_INCLUDE}) endif()
if(VULKAN_LIB) target_link_libraries(exynosx940complete PRIVATE ${VULKAN_LIB}) endif()

install(TARGETS exynosx940complete LIBRARY DESTINATION usr/lib)
install(DIRECTORY etc/ DESTINATION usr/etc)
install(FILES README.md LICENSE DESTINATION usr/share/doc/exynosx940)
