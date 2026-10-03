find_package(NanoTest REQUIRED)
find_package(EADataStructures REQUIRED)

nano_add_executable(ESIMDTests
        SOURCES ESIMDTests.cpp
        TARGETS ESIMD ea_data_structures)

add_executable(ESIMDBench ESIMDBench.cpp)
target_link_libraries(ESIMDBench PRIVATE ESIMD ea_data_structures)

# A benchmark is only meaningful optimized, so force it -- but only outside
# Debug: a Debug run is never representative, and skipping the call there keeps
# esimd_force_optimization's directory-scoped /RTC1 strip off ESIMDTests, which
# shares this directory and wants its Debug runtime checks intact.
if (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    esimd_force_optimization(ESIMDBench)
endif ()
