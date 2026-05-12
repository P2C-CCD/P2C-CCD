add_library(
  p2cccd_core
  common/validators.cpp
  common/group_scheduler.cpp
  geometry/mesh_io.cpp
  geometry/motion_utils.cpp
  geometry/patch_builder.cpp
  geometry/proxy.cpp
  rt_candidate/candidate_buffer.cpp
  rt_candidate/candidate_stats.cpp
  rt_candidate/cpu_reference_candidate_generator.cpp
  rt_candidate/candidate_generator.cpp
  rt_candidate/external_batch_candidate.cpp
  rt_candidate/optix_candidate_tracer.cpp
  rt_candidate/patch_granularity_tuning.cpp
  rt_candidate/proxy_scene.cpp
  certificate/certificate_engine.cpp
  certificate/mesh_exact_query.cpp
  proposal/proposal_features.cpp
  proposal/proposal_policy.cpp
)

target_include_directories(
  p2cccd_core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_features(p2cccd_core PUBLIC cxx_std_20)

if(P2CCCD_ENABLE_CUDA)
  target_sources(p2cccd_core PRIVATE certificate/certificate_cuda.cu)
  target_link_libraries(p2cccd_core PRIVATE CUDA::cudart)
  target_compile_definitions(p2cccd_core PUBLIC P2CCCD_HAS_CUDA=1)
else()
  target_sources(p2cccd_core PRIVATE certificate/certificate_cuda_stub.cpp)
  target_compile_definitions(p2cccd_core PUBLIC P2CCCD_HAS_CUDA=0)
endif()

if(P2CCCD_ENABLE_OPTIX)
  set(P2CCCD_OPTIX_CANDIDATE_PTX "${CMAKE_CURRENT_BINARY_DIR}/rt_candidate/optix_candidate_program.ptx")
  add_custom_command(
    OUTPUT "${P2CCCD_OPTIX_CANDIDATE_PTX}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/rt_candidate"
    COMMAND "${CMAKE_CUDA_COMPILER}"
            --ptx
            --std=c++17
            -I"${P2CCCD_OPTIX_INCLUDE_DIR}"
            -I"${CMAKE_CURRENT_SOURCE_DIR}"
            "${CMAKE_CURRENT_SOURCE_DIR}/rt_candidate/optix_candidate_program.cu"
            -o "${P2CCCD_OPTIX_CANDIDATE_PTX}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/rt_candidate/optix_candidate_program.cu"
    VERBATIM
  )
  add_custom_target(p2cccd_optix_candidate_ptx DEPENDS "${P2CCCD_OPTIX_CANDIDATE_PTX}")
  add_dependencies(p2cccd_core p2cccd_optix_candidate_ptx)
  target_include_directories(p2cccd_core PRIVATE ${P2CCCD_OPTIX_INCLUDE_DIR})
  target_link_libraries(p2cccd_core PRIVATE CUDA::cuda_driver CUDA::cudart)
  target_compile_definitions(
    p2cccd_core
    PUBLIC
      P2CCCD_HAS_OPTIX=1
    PRIVATE
      P2CCCD_OPTIX_CANDIDATE_PTX_PATH="${P2CCCD_OPTIX_CANDIDATE_PTX}"
  )
else()
  target_compile_definitions(p2cccd_core PUBLIC P2CCCD_HAS_OPTIX=0)
endif()

add_library(P2CCCD::core ALIAS p2cccd_core)

if(P2CCCD_ENABLE_PYTHON)
  find_package(pybind11 QUIET)
  if(NOT pybind11_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/lib/pybind11/CMakeLists.txt")
    add_subdirectory("${PROJECT_SOURCE_DIR}/lib/pybind11" "${PROJECT_BINARY_DIR}/lib/pybind11")
    set(pybind11_FOUND ON)
  endif()
  if(pybind11_FOUND)
    pybind11_add_module(p2cccd_cpp MODULE NO_EXTRAS bindings/py_module.cpp)
    target_link_libraries(p2cccd_cpp PRIVATE p2cccd_core)
    target_include_directories(p2cccd_cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    if(MSVC)
      target_compile_options(p2cccd_cpp PRIVATE /bigobj)
    endif()
  else()
    message(WARNING "P2CCCD_ENABLE_PYTHON is ON but pybind11 was not found.")
  endif()
endif()

if(P2CCCD_BUILD_TESTS)
  add_executable(
    p2cccd_contract_tests
    ${PROJECT_SOURCE_DIR}/tests/cpp/test_validators.cpp
  )
  target_link_libraries(p2cccd_contract_tests PRIVATE P2CCCD::core)
  add_test(NAME p2cccd_contract_tests COMMAND p2cccd_contract_tests)

  add_executable(
    p2cccd_geometry_tests
    ${PROJECT_SOURCE_DIR}/tests/cpp/test_geometry.cpp
  )
  target_link_libraries(p2cccd_geometry_tests PRIVATE P2CCCD::core)
  add_test(NAME p2cccd_geometry_tests COMMAND p2cccd_geometry_tests)

  add_executable(
    p2cccd_motion_proxy_tests
    ${PROJECT_SOURCE_DIR}/tests/cpp/test_motion_proxy.cpp
  )
  target_link_libraries(p2cccd_motion_proxy_tests PRIVATE P2CCCD::core)
  add_test(NAME p2cccd_motion_proxy_tests COMMAND p2cccd_motion_proxy_tests)

  add_executable(
    p2cccd_rt_candidate_tests
    ${PROJECT_SOURCE_DIR}/tests/cpp/test_rt_candidate.cpp
  )
  target_link_libraries(p2cccd_rt_candidate_tests PRIVATE P2CCCD::core)
  add_test(NAME p2cccd_rt_candidate_tests COMMAND p2cccd_rt_candidate_tests)

  add_executable(
    p2cccd_candidate_recall_test
    ${PROJECT_SOURCE_DIR}/tests/cpp/test_candidate_recall.cpp
  )
  target_link_libraries(p2cccd_candidate_recall_test PRIVATE P2CCCD::core)
  add_test(NAME candidate_recall_test COMMAND p2cccd_candidate_recall_test)

  add_executable(
    p2cccd_proxy_bound_unit_test
    ${PROJECT_SOURCE_DIR}/tests/cpp/test_proxy_bounds.cpp
  )
  target_link_libraries(p2cccd_proxy_bound_unit_test PRIVATE P2CCCD::core)
  add_test(NAME proxy_bound_unit_test COMMAND p2cccd_proxy_bound_unit_test)

  add_executable(
    p2cccd_slab_refinement_recall_test
    ${PROJECT_SOURCE_DIR}/tests/cpp/test_slab_refinement_recall.cpp
  )
  target_link_libraries(p2cccd_slab_refinement_recall_test PRIVATE P2CCCD::core)
  add_test(NAME slab_refinement_recall_test COMMAND p2cccd_slab_refinement_recall_test)

  add_executable(
    p2cccd_patch_granularity_tuning_test
    ${PROJECT_SOURCE_DIR}/tests/cpp/test_patch_granularity_tuning.cpp
  )
  target_link_libraries(p2cccd_patch_granularity_tuning_test PRIVATE P2CCCD::core)
  add_test(NAME patch_granularity_tuning_test COMMAND p2cccd_patch_granularity_tuning_test)

  add_executable(
    p2cccd_certificate_engine_test
    ${PROJECT_SOURCE_DIR}/tests/cpp/test_certificate_engine.cpp
  )
  target_link_libraries(p2cccd_certificate_engine_test PRIVATE P2CCCD::core)
  add_test(NAME certificate_engine_test COMMAND p2cccd_certificate_engine_test)

  add_executable(
    p2cccd_mesh_exact_query_test
    ${PROJECT_SOURCE_DIR}/tests/cpp/test_mesh_exact_query.cpp
  )
  target_link_libraries(p2cccd_mesh_exact_query_test PRIVATE P2CCCD::core)
  add_test(NAME mesh_exact_query_test COMMAND p2cccd_mesh_exact_query_test)

  add_executable(
    p2cccd_certificate_cuda_test
    ${PROJECT_SOURCE_DIR}/tests/cpp/test_certificate_cuda.cpp
  )
  target_link_libraries(p2cccd_certificate_cuda_test PRIVATE P2CCCD::core)
  add_test(NAME certificate_cuda_test COMMAND p2cccd_certificate_cuda_test)

  add_executable(
    p2cccd_proposal_features_test
    ${PROJECT_SOURCE_DIR}/tests/cpp/test_proposal_features.cpp
  )
  target_link_libraries(p2cccd_proposal_features_test PRIVATE P2CCCD::core)
  add_test(NAME proposal_features_test COMMAND p2cccd_proposal_features_test)

  add_executable(
    p2cccd_proposal_monotonicity_test
    ${PROJECT_SOURCE_DIR}/tests/cpp/test_proposal_monotonicity.cpp
  )
  target_link_libraries(p2cccd_proposal_monotonicity_test PRIVATE P2CCCD::core)
  add_test(NAME proposal_monotonicity_test COMMAND p2cccd_proposal_monotonicity_test)

  add_executable(
    p2cccd_queue_conservation_test
    ${PROJECT_SOURCE_DIR}/tests/cpp/test_queue_conservation.cpp
  )
  target_link_libraries(p2cccd_queue_conservation_test PRIVATE P2CCCD::core)
  add_test(NAME queue_conservation_test COMMAND p2cccd_queue_conservation_test)

  add_executable(
    p2cccd_proposal_ood_fallback_test
    ${PROJECT_SOURCE_DIR}/tests/cpp/test_proposal_ood_fallback.cpp
  )
  target_link_libraries(p2cccd_proposal_ood_fallback_test PRIVATE P2CCCD::core)
  add_test(NAME proposal_ood_fallback_test COMMAND p2cccd_proposal_ood_fallback_test)
endif()

if(P2CCCD_BUILD_TOOLS)
  add_executable(
    p2cccd_geometry_proxy_viewer
    ${PROJECT_SOURCE_DIR}/tools/geometry_proxy_viewer.cpp
  )
  target_link_libraries(p2cccd_geometry_proxy_viewer PRIVATE P2CCCD::core)

  add_executable(
    p2cccd_stage_overview_viewer
    ${PROJECT_SOURCE_DIR}/tools/stage_overview_viewer.cpp
  )
  target_link_libraries(p2cccd_stage_overview_viewer PRIVATE P2CCCD::core)

  add_executable(
    p2cccd_patch_granularity_tuner
    ${PROJECT_SOURCE_DIR}/tools/patch_granularity_tuner.cpp
  )
  target_link_libraries(p2cccd_patch_granularity_tuner PRIVATE P2CCCD::core)

  add_executable(
    p2cccd_stage4_certificate_viewer
    ${PROJECT_SOURCE_DIR}/tools/stage4_certificate_viewer.cpp
  )
  target_link_libraries(p2cccd_stage4_certificate_viewer PRIVATE P2CCCD::core)

  add_executable(
    p2cccd_proposal_queue_viewer
    ${PROJECT_SOURCE_DIR}/tools/proposal_queue_viewer.cpp
  )
  target_link_libraries(p2cccd_proposal_queue_viewer PRIVATE P2CCCD::core)
endif()
