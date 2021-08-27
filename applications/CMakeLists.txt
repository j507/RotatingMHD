PROJECT(applications)

INCLUDE_DIRECTORIES(
    include
    ${CMAKE_SOURCE_DIR}/include
    )

SET(SOURCE_FILES
#    Christensen.cc
    DFG.cc
#    MIT.cc
    step-35.cc
#    AdvectionDiffusion.cc
    Diffusion.cc
    DiffusionTest.cc
    )

FOREACH(sourcefile ${SOURCE_FILES})
    # string replace: cut off .cc from files
    STRING(REPLACE ".cc" "" executablename ${sourcefile} )
    PROJECT(${executablename})
    ADD_EXECUTABLE(${executablename} ${sourcefile})
    DEAL_II_SETUP_TARGET(${executablename})
    TARGET_LINK_LIBRARIES(${executablename} rotatingMHD stdc++fs)
ENDFOREACH(sourcefile ${APP_SOURCES})
