#pragma once

#include "common/status.h"
#include "geometry/mesh.h"

#include <filesystem>

namespace p2cccd {

struct ObjLoadOptions {
  bool triangulate_polygon_faces = true;
  bool use_object_groups_as_patch_ids = false;
};

enum class MeshFileFormat {
  kObj,
  kStl,
};

Status LoadTriangleMeshObj(const std::filesystem::path& path,
                           Mesh* mesh,
                           ObjLoadOptions options = {});
Status LoadTriangleMesh(const std::filesystem::path& path,
                        Mesh* mesh,
                        ObjLoadOptions obj_options = {});
Status ValidateTriangleMesh(const Mesh& mesh);
Status CenterMeshAtAabbCenter(const Mesh& input,
                              Mesh* centered,
                              std::array<double, 3>* original_center = nullptr);

}  // namespace p2cccd
