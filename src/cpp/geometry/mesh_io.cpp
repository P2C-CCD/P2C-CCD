#include "geometry/mesh_io.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace p2cccd {
namespace {

Status RequireMeshOutput(Mesh* mesh) {
  if (mesh == nullptr) {
    return Status::Error("mesh output pointer is null");
  }
  return Status::Ok();
}

struct VertexKey {
  float x = 0.0F;
  float y = 0.0F;
  float z = 0.0F;

  bool operator==(const VertexKey& other) const {
    return x == other.x && y == other.y && z == other.z;
  }
};

struct VertexKeyHash {
  std::size_t operator()(const VertexKey& key) const {
    const std::size_t hx = std::hash<float>{}(key.x);
    const std::size_t hy = std::hash<float>{}(key.y);
    const std::size_t hz = std::hash<float>{}(key.z);
    return hx ^ (hy << 1U) ^ (hz << 2U);
  }
};

std::string StripComment(const std::string& line) {
  const std::size_t comment_pos = line.find('#');
  if (comment_pos == std::string::npos) {
    return line;
  }
  return line.substr(0, comment_pos);
}

Status ParseObjVertexIndex(const std::string& token,
                           std::size_t vertex_count,
                           std::uint32_t* vertex_index) {
  if (vertex_index == nullptr) {
    return Status::Error("vertex index output pointer is null");
  }
  const std::size_t slash_pos = token.find('/');
  const std::string index_text = token.substr(0, slash_pos);
  if (index_text.empty()) {
    return Status::Error("OBJ face token has empty vertex index");
  }

  std::size_t parsed_chars = 0;
  long long obj_index = 0;
  try {
    obj_index = std::stoll(index_text, &parsed_chars, 10);
  } catch (const std::exception&) {
    return Status::Error("OBJ face token has invalid vertex index");
  }
  if (parsed_chars != index_text.size() || obj_index == 0) {
    return Status::Error("OBJ vertex index must be a non-zero integer");
  }

  long long zero_based = -1;
  if (obj_index > 0) {
    zero_based = obj_index - 1;
  } else {
    zero_based = static_cast<long long>(vertex_count) + obj_index;
  }

  if (zero_based < 0 || static_cast<std::size_t>(zero_based) >= vertex_count) {
    return Status::Error("OBJ face references a vertex outside the loaded vertex range");
  }
  if (zero_based > static_cast<long long>(std::numeric_limits<std::uint32_t>::max())) {
    return Status::Error("OBJ vertex index exceeds uint32 range");
  }

  *vertex_index = static_cast<std::uint32_t>(zero_based);
  return Status::Ok();
}

std::array<double, 3> Sub(std::array<double, 3> lhs, std::array<double, 3> rhs) {
  return {lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2]};
}

std::array<double, 3> Cross(std::array<double, 3> lhs, std::array<double, 3> rhs) {
  return {
      lhs[1] * rhs[2] - lhs[2] * rhs[1],
      lhs[2] * rhs[0] - lhs[0] * rhs[2],
      lhs[0] * rhs[1] - lhs[1] * rhs[0],
  };
}

double Dot(std::array<double, 3> lhs, std::array<double, 3> rhs) {
  return lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2];
}

double SquaredNorm(std::array<double, 3> value) {
  return Dot(value, value);
}

double TriangleAreaSquared(const Mesh& mesh, const std::array<std::uint32_t, 3>& triangle) {
  const std::array<double, 3> edge0 =
      Sub(mesh.vertices_ref[triangle[1]], mesh.vertices_ref[triangle[0]]);
  const std::array<double, 3> edge1 =
      Sub(mesh.vertices_ref[triangle[2]], mesh.vertices_ref[triangle[0]]);
  return 0.25 * SquaredNorm(Cross(edge0, edge1));
}

Status AddFaceTriangles(const std::vector<std::uint32_t>& face,
                        std::uint32_t patch_id,
                        bool triangulate_polygon_faces,
                        Mesh* mesh) {
  if (face.size() < 3) {
    return Status::Error("OBJ face must have at least 3 vertices");
  }
  if (face.size() > 3 && !triangulate_polygon_faces) {
    return Status::Error("OBJ polygon face found but triangulation is disabled");
  }

  for (std::size_t i = 1; i + 1 < face.size(); ++i) {
    if (face[0] == face[i] || face[0] == face[i + 1] || face[i] == face[i + 1]) {
      return Status::Error("OBJ face creates a degenerate triangle with duplicate indices");
    }
    mesh->triangles.push_back({face[0], face[i], face[i + 1]});
    mesh->patch_ids.push_back(patch_id);
  }
  return Status::Ok();
}

bool LooksLikeBinaryStl(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return false;
  }

  input.seekg(0, std::ios::end);
  const std::streamoff file_size = input.tellg();
  if (file_size < 84) {
    return false;
  }
  input.seekg(80, std::ios::beg);
  std::uint32_t triangle_count = 0;
  input.read(reinterpret_cast<char*>(&triangle_count), sizeof(triangle_count));
  if (!input) {
    return false;
  }
  const std::uint64_t expected_size =
      84ULL + static_cast<std::uint64_t>(triangle_count) * 50ULL;
  return static_cast<std::uint64_t>(file_size) == expected_size;
}

Status AddOrGetVertexIndex(const VertexKey& key,
                           std::unordered_map<VertexKey, std::uint32_t, VertexKeyHash>* vertex_map,
                           Mesh* mesh,
                           std::uint32_t* vertex_index) {
  if (vertex_map == nullptr || mesh == nullptr || vertex_index == nullptr) {
    return Status::Error("STL vertex map arguments are null");
  }

  const auto found = vertex_map->find(key);
  if (found != vertex_map->end()) {
    *vertex_index = found->second;
    return Status::Ok();
  }

  if (mesh->vertices_ref.size() >=
      static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
    return Status::Error("STL vertex count exceeds uint32 range");
  }
  const std::uint32_t new_index = static_cast<std::uint32_t>(mesh->vertices_ref.size());
  mesh->vertices_ref.push_back(
      {static_cast<double>(key.x), static_cast<double>(key.y), static_cast<double>(key.z)});
  vertex_map->emplace(key, new_index);
  *vertex_index = new_index;
  return Status::Ok();
}

Status LoadTriangleMeshBinaryStl(const std::filesystem::path& path, Mesh* mesh) {
  if (auto status = RequireMeshOutput(mesh); !status.ok) {
    return status;
  }

  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return Status::Error("failed to open STL file: " + path.string());
  }

  std::array<char, 80> header{};
  input.read(header.data(), static_cast<std::streamsize>(header.size()));
  std::uint32_t triangle_count = 0;
  input.read(reinterpret_cast<char*>(&triangle_count), sizeof(triangle_count));
  if (!input) {
    return Status::Error("failed to read binary STL header: " + path.string());
  }

  Mesh loaded;
  loaded.triangles.reserve(triangle_count);
  std::unordered_map<VertexKey, std::uint32_t, VertexKeyHash> vertex_map;
  vertex_map.reserve(static_cast<std::size_t>(triangle_count) * 3U);

  for (std::uint32_t triangle_index = 0; triangle_index < triangle_count; ++triangle_index) {
    float values[12]{};
    input.read(reinterpret_cast<char*>(values), sizeof(values));
    std::uint16_t attribute_bytes = 0;
    input.read(reinterpret_cast<char*>(&attribute_bytes), sizeof(attribute_bytes));
    if (!input) {
      return Status::Error("unexpected EOF while reading binary STL triangle data");
    }

    std::array<std::uint32_t, 3> triangle{};
    for (std::size_t vertex = 0; vertex < 3; ++vertex) {
      const VertexKey key{
          values[3U + vertex * 3U + 0U],
          values[3U + vertex * 3U + 1U],
          values[3U + vertex * 3U + 2U],
      };
      if (auto status =
              AddOrGetVertexIndex(key, &vertex_map, &loaded, &triangle[vertex]);
          !status.ok) {
        return status;
      }
    }
    if (triangle[0] == triangle[1] || triangle[0] == triangle[2] || triangle[1] == triangle[2]) {
      continue;
    }
    loaded.triangles.push_back(triangle);
  }

  if (auto status = ValidateTriangleMesh(loaded); !status.ok) {
    return status;
  }
  *mesh = std::move(loaded);
  return Status::Ok();
}

Status LoadTriangleMeshAsciiStl(const std::filesystem::path& path, Mesh* mesh) {
  if (auto status = RequireMeshOutput(mesh); !status.ok) {
    return status;
  }

  std::ifstream input(path);
  if (!input) {
    return Status::Error("failed to open STL file: " + path.string());
  }

  Mesh loaded;
  std::unordered_map<VertexKey, std::uint32_t, VertexKeyHash> vertex_map;
  std::vector<std::uint32_t> current_triangle;
  current_triangle.reserve(3);

  std::string line;
  while (std::getline(input, line)) {
    std::istringstream stream(line);
    std::string tag;
    if (!(stream >> tag)) {
      continue;
    }
    if (tag != "vertex" && tag != "VERTEX") {
      continue;
    }

    VertexKey key{};
    if (!(stream >> key.x >> key.y >> key.z)) {
      return Status::Error("invalid ASCII STL vertex row in " + path.string());
    }
    std::uint32_t vertex_index = 0;
    if (auto status = AddOrGetVertexIndex(key, &vertex_map, &loaded, &vertex_index); !status.ok) {
      return status;
    }
    current_triangle.push_back(vertex_index);
    if (current_triangle.size() == 3U) {
      if (current_triangle[0] != current_triangle[1] &&
          current_triangle[0] != current_triangle[2] &&
          current_triangle[1] != current_triangle[2]) {
        loaded.triangles.push_back(
            {current_triangle[0], current_triangle[1], current_triangle[2]});
      }
      current_triangle.clear();
    }
  }

  if (!current_triangle.empty()) {
    return Status::Error("ASCII STL ended with a partial triangle");
  }
  if (auto status = ValidateTriangleMesh(loaded); !status.ok) {
    return status;
  }
  *mesh = std::move(loaded);
  return Status::Ok();
}

}  // namespace

Status ValidateTriangleMesh(const Mesh& mesh) {
  if (mesh.vertices_ref.empty()) {
    return Status::Error("mesh has no vertices");
  }
  if (mesh.triangles.empty()) {
    return Status::Error("mesh has no triangles");
  }
  if (!mesh.patch_ids.empty() && mesh.patch_ids.size() != mesh.triangles.size()) {
    return Status::Error("mesh patch_ids must be empty or match triangle count");
  }

  for (const auto& vertex : mesh.vertices_ref) {
    for (double value : vertex) {
      if (!std::isfinite(value)) {
        return Status::Error("mesh vertex contains a non-finite coordinate");
      }
    }
  }

  for (const auto& triangle : mesh.triangles) {
    if (triangle[0] == triangle[1] || triangle[0] == triangle[2] || triangle[1] == triangle[2]) {
      return Status::Error("mesh triangle contains duplicate vertex indices");
    }
    for (std::uint32_t index : triangle) {
      if (index >= mesh.vertices_ref.size()) {
        return Status::Error("mesh triangle references a vertex outside the vertex array");
      }
    }
    if (TriangleAreaSquared(mesh, triangle) <= 0.0) {
      return Status::Error("mesh triangle has zero area");
    }
  }

  return Status::Ok();
}

Status LoadTriangleMeshObj(const std::filesystem::path& path, Mesh* mesh, ObjLoadOptions options) {
  if (auto status = RequireMeshOutput(mesh); !status.ok) {
    return status;
  }

  std::ifstream input(path);
  if (!input) {
    return Status::Error("failed to open OBJ file: " + path.string());
  }

  Mesh loaded;
  std::string line;
  std::uint32_t current_patch_id = 0;
  std::uint32_t next_patch_id = 1;
  std::uint64_t line_number = 0;

  while (std::getline(input, line)) {
    ++line_number;
    std::istringstream stream(StripComment(line));
    std::string tag;
    if (!(stream >> tag)) {
      continue;
    }

    if (tag == "v") {
      std::array<double, 3> vertex{};
      if (!(stream >> vertex[0] >> vertex[1] >> vertex[2])) {
        return Status::Error("invalid OBJ vertex at line " + std::to_string(line_number));
      }
      loaded.vertices_ref.push_back(vertex);
      continue;
    }

    if (tag == "f") {
      std::vector<std::uint32_t> face;
      std::string token;
      while (stream >> token) {
        std::uint32_t vertex_index = 0;
        if (auto status =
                ParseObjVertexIndex(token, loaded.vertices_ref.size(), &vertex_index);
            !status.ok) {
          return Status::Error(status.message + " at line " + std::to_string(line_number));
        }
        face.push_back(vertex_index);
      }
      if (auto status = AddFaceTriangles(face,
                                         current_patch_id,
                                         options.triangulate_polygon_faces,
                                         &loaded);
          !status.ok) {
        return Status::Error(status.message + " at line " + std::to_string(line_number));
      }
      continue;
    }

    if ((tag == "g" || tag == "o") && options.use_object_groups_as_patch_ids) {
      current_patch_id = next_patch_id++;
      continue;
    }
  }

  if (auto status = ValidateTriangleMesh(loaded); !status.ok) {
    return status;
  }

  *mesh = std::move(loaded);
  return Status::Ok();
}

Status LoadTriangleMesh(const std::filesystem::path& path, Mesh* mesh, ObjLoadOptions obj_options) {
  const std::string extension = path.extension().string();
  if (extension == ".obj" || extension == ".OBJ") {
    return LoadTriangleMeshObj(path, mesh, obj_options);
  }
  if (extension == ".stl" || extension == ".STL") {
    return LooksLikeBinaryStl(path) ? LoadTriangleMeshBinaryStl(path, mesh)
                                    : LoadTriangleMeshAsciiStl(path, mesh);
  }
  return Status::Error("unsupported triangle mesh format: " + extension);
}

Status CenterMeshAtAabbCenter(const Mesh& input,
                              Mesh* centered,
                              std::array<double, 3>* original_center) {
  if (centered == nullptr) {
    return Status::Error("centered mesh output pointer is null");
  }
  if (auto status = ValidateTriangleMesh(input); !status.ok) {
    return status;
  }

  std::array<double, 3> bounds_min = input.vertices_ref.front();
  std::array<double, 3> bounds_max = input.vertices_ref.front();
  for (const auto& vertex : input.vertices_ref) {
    for (std::size_t axis = 0; axis < 3; ++axis) {
      bounds_min[axis] = std::min(bounds_min[axis], vertex[axis]);
      bounds_max[axis] = std::max(bounds_max[axis], vertex[axis]);
    }
  }
  const std::array<double, 3> center{
      0.5 * (bounds_min[0] + bounds_max[0]),
      0.5 * (bounds_min[1] + bounds_max[1]),
      0.5 * (bounds_min[2] + bounds_max[2]),
  };

  Mesh output = input;
  for (auto& vertex : output.vertices_ref) {
    vertex = Sub(vertex, center);
  }
  if (original_center != nullptr) {
    *original_center = center;
  }
  *centered = std::move(output);
  return Status::Ok();
}

}  // namespace p2cccd
