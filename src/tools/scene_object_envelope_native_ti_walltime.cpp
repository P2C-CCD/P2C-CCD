#include <tight_inclusion/ccd.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Args {
  fs::path frame0;
  fs::path frame1;
  fs::path output_jsonl;
  std::string scene = "scene";
  double thickness = 0.0;
  double ms = 0.0;
  double tolerance = 1.0e-6;
  double t_max = 1.0;
  long max_itr = 1000000;
  std::uint64_t max_vf_candidates = 0;
  std::uint64_t max_ee_candidates = 0;
  fs::path stpf_weights;
  std::size_t proposal_top_k = 4096;
  std::size_t optimized_frontier_k = 1024;
  std::uint64_t optimized_scan_limit_per_group = 1048576;
  int optimized_random_gate_object_count = 128;
  fs::path feature_jsonl;
  std::uint64_t feature_negative_stride = 0;
  bool feature_export_only = false;
  bool exclude_self_object_pairs = false;
};

struct Vec3 {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
};

struct Aabb {
  Vec3 mn{
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity(),
  };
  Vec3 mx{
      -std::numeric_limits<double>::infinity(),
      -std::numeric_limits<double>::infinity(),
      -std::numeric_limits<double>::infinity(),
  };
};

struct Face {
  int v[3] = {0, 0, 0};
  int object_id = -1;
  Aabb box;
};

struct Edge {
  int a = 0;
  int b = 0;
  int object_id = -1;
  Aabb box;
};

struct VertexProxy {
  int id = 0;
  int object_id = -1;
  Aabb box;
};

struct Candidate {
  int a = 0;
  int b = 0;
};

struct Mesh {
  std::vector<Vec3> vertices;
  std::vector<Face> faces;
};

struct Dsu {
  std::vector<int> parent;
  std::vector<unsigned char> rank;

  explicit Dsu(std::size_t n) : parent(n), rank(n, 0) {
    std::iota(parent.begin(), parent.end(), 0);
  }

  int find(int x) {
    while (parent[x] != x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  }

  void unite(int a, int b) {
    int ra = find(a);
    int rb = find(b);
    if (ra == rb) {
      return;
    }
    if (rank[ra] < rank[rb]) {
      parent[ra] = rb;
    } else if (rank[ra] > rank[rb]) {
      parent[rb] = ra;
    } else {
      parent[rb] = ra;
      ++rank[ra];
    }
  }
};

struct PreparedScene {
  Mesh mesh0;
  Mesh mesh1;
  std::vector<int> vertex_object;
  std::vector<Aabb> object_boxes;
  std::vector<unsigned char> object_pair_allowed;
  std::vector<VertexProxy> vertices;
  std::vector<Edge> edges;
  int object_count = 0;
  int object_envelope_count = 0;
  bool exclude_self_object_pairs = false;
};

struct MethodRow {
  std::string method;
  std::string kind;
  std::uint64_t candidates = 0;
  std::uint64_t exact_calls = 0;
  std::uint64_t proposal_exact_calls = 0;
  std::uint64_t fallback_exact_calls = 0;
  std::uint64_t positive_exact_calls = 0;
  std::uint64_t negative_exact_calls = 0;
  std::uint64_t positive_proposal_exact_calls = 0;
  std::uint64_t positive_fallback_exact_calls = 0;
  std::uint64_t negative_proposal_exact_calls = 0;
  std::uint64_t negative_fallback_exact_calls = 0;
  std::uint64_t positive_proposal_hits = 0;
  std::uint64_t positive_fallback_hits = 0;
  std::uint64_t positive_first_hit_rank_sum = 0;
  std::uint64_t positive_first_hit_groups = 0;
  std::uint64_t hit_count = 0;
  std::uint64_t first_hit_rank = 0;
  std::uint64_t group_count = 0;
  std::uint64_t positive_groups = 0;
  std::uint64_t negative_groups = 0;
  std::uint64_t fallback_groups = 0;
  std::uint64_t tp = 0;
  std::uint64_t tn = 0;
  std::uint64_t fp = 0;
  std::uint64_t fn = 0;
  bool detected_hit = false;
  bool capped = false;
  bool random_gate_used = false;
  std::uint64_t effective_frontier_k = 0;
  std::uint64_t effective_proposal_k = 0;
  double envelope_ms = 0.0;
  double ordering_ms = 0.0;
  double exact_ms = 0.0;
  double total_ms = 0.0;
};

struct TinyStpfWeights {
  std::array<double, 32> mean{};
  std::array<double, 32> std{};
  std::vector<double> w0;
  std::vector<double> b0;
  std::vector<double> w1;
  std::vector<double> b1;
  std::vector<double> w2;
  double b2 = 0.0;
  bool loaded = false;
};

struct GroupTruth {
  bool positive = false;
};

struct ScoredCandidate {
  double score = 0.0;
  std::uint64_t index = 0;
};

struct FastSignals {
  double inv_gap = 0.0;
  double rel_motion = 0.0;
  double overlap = 0.0;
  double compactness = 0.0;
};

struct LearnedFrontierCandidate {
  std::uint64_t index = 0;
  double learned = 0.0;
  double proximity = 0.0;
  double motion = 0.0;
  double density = 0.0;
  double score = 0.0;
};

Vec3 Min(const Vec3& a, const Vec3& b) {
  return {std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z)};
}

Vec3 Max(const Vec3& a, const Vec3& b) {
  return {std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z)};
}

Vec3 Add(const Vec3& a, const Vec3& b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}

Vec3 Sub(const Vec3& a, const Vec3& b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Vec3 Scale(const Vec3& a, double s) {
  return {a.x * s, a.y * s, a.z * s};
}

double Dot(const Vec3& a, const Vec3& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3 Cross(const Vec3& a, const Vec3& b) {
  return {
      a.y * b.z - a.z * b.y,
      a.z * b.x - a.x * b.z,
      a.x * b.y - a.y * b.x,
  };
}

double Norm(const Vec3& a) {
  return std::sqrt(Dot(a, a));
}

void Extend(Aabb* box, const Vec3& p) {
  box->mn = Min(box->mn, p);
  box->mx = Max(box->mx, p);
}

void Expand(Aabb* box, double delta) {
  box->mn.x -= delta;
  box->mn.y -= delta;
  box->mn.z -= delta;
  box->mx.x += delta;
  box->mx.y += delta;
  box->mx.z += delta;
}

bool Overlap(const Aabb& a, const Aabb& b) {
  return a.mn.x <= b.mx.x && b.mn.x <= a.mx.x &&
         a.mn.y <= b.mx.y && b.mn.y <= a.mx.y &&
         a.mn.z <= b.mx.z && b.mn.z <= a.mx.z;
}

Aabb SweptVertexBox(const std::vector<Vec3>& v0, const std::vector<Vec3>& v1, int index, double thickness) {
  Aabb box;
  Extend(&box, v0[index]);
  Extend(&box, v1[index]);
  Expand(&box, thickness);
  return box;
}

Aabb SweptFaceBox(const std::vector<Vec3>& v0, const std::vector<Vec3>& v1, const Face& face, double thickness) {
  Aabb box;
  for (int k = 0; k < 3; ++k) {
    Extend(&box, v0[face.v[k]]);
    Extend(&box, v1[face.v[k]]);
  }
  Expand(&box, thickness);
  return box;
}

Aabb SweptEdgeBox(const std::vector<Vec3>& v0, const std::vector<Vec3>& v1, int a, int b, double thickness) {
  Aabb box;
  Extend(&box, v0[a]);
  Extend(&box, v0[b]);
  Extend(&box, v1[a]);
  Extend(&box, v1[b]);
  Expand(&box, thickness);
  return box;
}

std::string Trim(std::string value) {
  value.erase(value.begin(), std::find_if(value.begin(), value.end(), [](unsigned char ch) {
    return !std::isspace(ch);
  }));
  value.erase(std::find_if(value.rbegin(), value.rend(), [](unsigned char ch) {
    return !std::isspace(ch);
  }).base(), value.end());
  return value;
}

std::vector<std::string> SplitWords(const std::string& line) {
  std::istringstream stream(line);
  std::vector<std::string> out;
  std::string token;
  while (stream >> token) {
    out.push_back(token);
  }
  return out;
}

template <typename T>
T ByteSwap(T value) {
  std::array<unsigned char, sizeof(T)> bytes{};
  std::memcpy(bytes.data(), &value, sizeof(T));
  std::reverse(bytes.begin(), bytes.end());
  std::memcpy(&value, bytes.data(), sizeof(T));
  return value;
}

template <typename T>
T ReadScalar(std::istream& in, bool big_endian) {
  T value{};
  in.read(reinterpret_cast<char*>(&value), sizeof(T));
  if (!in) {
    throw std::runtime_error("unexpected EOF while reading binary PLY");
  }
  if (big_endian && sizeof(T) > 1) {
    value = ByteSwap(value);
  }
  return value;
}

double ReadTypedScalar(std::istream& in, const std::string& type, bool big_endian) {
  if (type == "double" || type == "float64") {
    return ReadScalar<double>(in, big_endian);
  }
  if (type == "float" || type == "float32") {
    return static_cast<double>(ReadScalar<float>(in, big_endian));
  }
  if (type == "int" || type == "int32") {
    return static_cast<double>(ReadScalar<std::int32_t>(in, big_endian));
  }
  if (type == "uint" || type == "uint32") {
    return static_cast<double>(ReadScalar<std::uint32_t>(in, big_endian));
  }
  if (type == "short" || type == "int16") {
    return static_cast<double>(ReadScalar<std::int16_t>(in, big_endian));
  }
  if (type == "ushort" || type == "uint16") {
    return static_cast<double>(ReadScalar<std::uint16_t>(in, big_endian));
  }
  if (type == "uchar" || type == "uint8") {
    return static_cast<double>(ReadScalar<std::uint8_t>(in, big_endian));
  }
  if (type == "char" || type == "int8") {
    return static_cast<double>(ReadScalar<std::int8_t>(in, big_endian));
  }
  throw std::runtime_error("unsupported PLY scalar type: " + type);
}

std::uint32_t ReadTypedUInt(std::istream& in, const std::string& type, bool big_endian) {
  if (type == "uchar" || type == "uint8") {
    return ReadScalar<std::uint8_t>(in, big_endian);
  }
  if (type == "char" || type == "int8") {
    return static_cast<std::uint32_t>(ReadScalar<std::int8_t>(in, big_endian));
  }
  if (type == "ushort" || type == "uint16") {
    return ReadScalar<std::uint16_t>(in, big_endian);
  }
  if (type == "short" || type == "int16") {
    return static_cast<std::uint32_t>(ReadScalar<std::int16_t>(in, big_endian));
  }
  if (type == "uint" || type == "uint32") {
    return ReadScalar<std::uint32_t>(in, big_endian);
  }
  if (type == "int" || type == "int32") {
    return static_cast<std::uint32_t>(ReadScalar<std::int32_t>(in, big_endian));
  }
  throw std::runtime_error("unsupported PLY integer type: " + type);
}

Mesh LoadBinaryPly(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to open PLY " + path.string());
  }

  std::string line;
  std::getline(in, line);
  if (Trim(line) != "ply") {
    throw std::runtime_error("not a PLY file: " + path.string());
  }

  bool binary_little = false;
  bool binary_big = false;
  std::size_t vertex_count = 0;
  std::size_t face_count = 0;
  std::string current_element;
  std::vector<std::pair<std::string, std::string>> vertex_props;
  std::string face_count_type = "uint8";
  std::string face_index_type = "int32";

  while (std::getline(in, line)) {
    line = Trim(line);
    if (line == "end_header") {
      break;
    }
    const auto words = SplitWords(line);
    if (words.empty()) {
      continue;
    }
    if (words[0] == "format") {
      binary_little = words.size() >= 2 && words[1] == "binary_little_endian";
      binary_big = words.size() >= 2 && words[1] == "binary_big_endian";
    } else if (words[0] == "element" && words.size() >= 3) {
      current_element = words[1];
      if (current_element == "vertex") {
        vertex_count = static_cast<std::size_t>(std::stoull(words[2]));
      } else if (current_element == "face") {
        face_count = static_cast<std::size_t>(std::stoull(words[2]));
      }
    } else if (words[0] == "property" && current_element == "vertex" && words.size() >= 3) {
      vertex_props.push_back({words[2], words[1]});
    } else if (words[0] == "property" && current_element == "face" && words.size() >= 5 &&
               words[1] == "list") {
      face_count_type = words[2];
      face_index_type = words[3];
    }
  }
  if (!binary_little && !binary_big) {
    throw std::runtime_error("only binary PLY is supported: " + path.string());
  }
  if (vertex_count == 0 || face_count == 0) {
    throw std::runtime_error("PLY has no vertices or faces: " + path.string());
  }

  Mesh mesh;
  mesh.vertices.resize(vertex_count);
  for (std::size_t i = 0; i < vertex_count; ++i) {
    Vec3 v;
    for (const auto& [name, type] : vertex_props) {
      const double value = ReadTypedScalar(in, type, binary_big);
      if (name == "x") {
        v.x = value;
      } else if (name == "y") {
        v.y = value;
      } else if (name == "z") {
        v.z = value;
      }
    }
    mesh.vertices[i] = v;
  }

  mesh.faces.reserve(face_count);
  for (std::size_t i = 0; i < face_count; ++i) {
    const std::uint32_t count = ReadTypedUInt(in, face_count_type, binary_big);
    std::vector<int> indices(count);
    for (std::uint32_t k = 0; k < count; ++k) {
      indices[k] = static_cast<int>(ReadTypedUInt(in, face_index_type, binary_big));
    }
    if (count < 3) {
      continue;
    }
    for (std::uint32_t k = 1; k + 1 < count; ++k) {
      Face face;
      face.v[0] = indices[0];
      face.v[1] = indices[k];
      face.v[2] = indices[k + 1];
      if (face.v[0] != face.v[1] && face.v[1] != face.v[2] && face.v[0] != face.v[2]) {
        mesh.faces.push_back(face);
      }
    }
  }
  return mesh;
}

std::uint64_t EdgeKey(int a, int b) {
  const std::uint32_t lo = static_cast<std::uint32_t>(std::min(a, b));
  const std::uint32_t hi = static_cast<std::uint32_t>(std::max(a, b));
  return (static_cast<std::uint64_t>(lo) << 32) | hi;
}

PreparedScene PrepareScene(const fs::path& frame0, const fs::path& frame1, double thickness, bool exclude_self_object_pairs) {
  PreparedScene scene;
  scene.exclude_self_object_pairs = exclude_self_object_pairs;
  scene.mesh0 = LoadBinaryPly(frame0);
  scene.mesh1 = LoadBinaryPly(frame1);
  if (scene.mesh0.vertices.size() != scene.mesh1.vertices.size()) {
    throw std::runtime_error("frame vertex counts differ");
  }
  if (scene.mesh0.faces.size() != scene.mesh1.faces.size()) {
    throw std::runtime_error("frame face counts differ");
  }

  const int n_vertices = static_cast<int>(scene.mesh0.vertices.size());
  Dsu dsu(scene.mesh0.vertices.size());
  for (const Face& face : scene.mesh0.faces) {
    dsu.unite(face.v[0], face.v[1]);
    dsu.unite(face.v[1], face.v[2]);
  }

  std::unordered_map<int, int> root_to_object;
  scene.vertex_object.assign(scene.mesh0.vertices.size(), -1);
  for (Face& face : scene.mesh0.faces) {
    const int root = dsu.find(face.v[0]);
    auto [it, inserted] = root_to_object.emplace(root, static_cast<int>(root_to_object.size()));
    face.object_id = it->second;
    for (int k = 0; k < 3; ++k) {
      scene.vertex_object[face.v[k]] = face.object_id;
    }
  }
  scene.object_count = static_cast<int>(root_to_object.size());
  scene.object_boxes.resize(scene.object_count);

  for (int v = 0; v < n_vertices; ++v) {
    const int obj = scene.vertex_object[v];
    if (obj < 0) {
      continue;
    }
    Extend(&scene.object_boxes[obj], scene.mesh0.vertices[v]);
    Extend(&scene.object_boxes[obj], scene.mesh1.vertices[v]);
  }
  for (Aabb& box : scene.object_boxes) {
    Expand(&box, thickness);
  }

  scene.object_pair_allowed.assign(scene.object_count * scene.object_count, 0);
  for (int a = 0; a < scene.object_count; ++a) {
    for (int b = a; b < scene.object_count; ++b) {
      if ((a == b && !exclude_self_object_pairs) || (a != b && Overlap(scene.object_boxes[a], scene.object_boxes[b]))) {
        scene.object_pair_allowed[a * scene.object_count + b] = 1;
        scene.object_pair_allowed[b * scene.object_count + a] = 1;
        ++scene.object_envelope_count;
      }
    }
  }

  scene.vertices.reserve(scene.mesh0.vertices.size());
  for (int v = 0; v < n_vertices; ++v) {
    if (scene.vertex_object[v] < 0) {
      continue;
    }
    scene.vertices.push_back(VertexProxy{v, scene.vertex_object[v],
                                         SweptVertexBox(scene.mesh0.vertices, scene.mesh1.vertices, v, thickness)});
  }

  for (Face& face : scene.mesh0.faces) {
    face.box = SweptFaceBox(scene.mesh0.vertices, scene.mesh1.vertices, face, thickness);
  }

  std::unordered_map<std::uint64_t, int> edge_map;
  edge_map.reserve(scene.mesh0.faces.size() * 2);
  for (const Face& face : scene.mesh0.faces) {
    for (int k = 0; k < 3; ++k) {
      const int a = face.v[k];
      const int b = face.v[(k + 1) % 3];
      const std::uint64_t key = EdgeKey(a, b);
      if (edge_map.find(key) != edge_map.end()) {
        continue;
      }
      Edge edge;
      edge.a = std::min(a, b);
      edge.b = std::max(a, b);
      edge.object_id = face.object_id;
      edge.box = SweptEdgeBox(scene.mesh0.vertices, scene.mesh1.vertices, edge.a, edge.b, thickness);
      edge_map.emplace(key, static_cast<int>(scene.edges.size()));
      scene.edges.push_back(edge);
    }
  }

  return scene;
}

bool ObjectPairAllowed(const PreparedScene& scene, int a, int b) {
  if (a < 0 || b < 0 || a >= scene.object_count || b >= scene.object_count) {
    return false;
  }
  return scene.object_pair_allowed[a * scene.object_count + b] != 0;
}

bool FaceContainsVertex(const Face& face, int vertex) {
  return face.v[0] == vertex || face.v[1] == vertex || face.v[2] == vertex;
}

bool EdgesShareVertex(const Edge& a, const Edge& b) {
  return a.a == b.a || a.a == b.b || a.b == b.a || a.b == b.b;
}

template <typename A, typename B, typename Callback>
bool SweepPairs(const std::vector<A>& lhs,
                const std::vector<B>& rhs,
                std::uint64_t max_candidates,
                Callback callback) {
  std::vector<int> lhs_order(lhs.size());
  std::vector<int> rhs_order(rhs.size());
  std::iota(lhs_order.begin(), lhs_order.end(), 0);
  std::iota(rhs_order.begin(), rhs_order.end(), 0);
  std::stable_sort(lhs_order.begin(), lhs_order.end(), [&](int a, int b) {
    return lhs[a].box.mn.x < lhs[b].box.mn.x;
  });
  std::stable_sort(rhs_order.begin(), rhs_order.end(), [&](int a, int b) {
    return rhs[a].box.mn.x < rhs[b].box.mn.x;
  });

  std::vector<int> active;
  active.reserve(std::min<std::size_t>(rhs.size(), 4096));
  std::size_t cursor = 0;
  std::uint64_t accepted = 0;
  for (int ai : lhs_order) {
    const Aabb& a_box = lhs[ai].box;
    while (cursor < rhs_order.size() && rhs[rhs_order[cursor]].box.mn.x <= a_box.mx.x) {
      active.push_back(rhs_order[cursor]);
      ++cursor;
    }
    std::size_t write = 0;
    for (int bi : active) {
      if (rhs[bi].box.mx.x >= a_box.mn.x) {
        active[write++] = bi;
      }
    }
    active.resize(write);
    for (int bi : active) {
      if (!Overlap(a_box, rhs[bi].box)) {
        continue;
      }
      if (callback(ai, bi)) {
        ++accepted;
        if (max_candidates != 0 && accepted >= max_candidates) {
          return true;
        }
      }
    }
  }
  return false;
}

std::vector<Candidate> BuildVFCandidates(const PreparedScene& scene,
                                         std::uint64_t max_candidates,
                                         bool* capped) {
  std::vector<Candidate> candidates;
  if (max_candidates != 0) {
    candidates.reserve(static_cast<std::size_t>(std::min<std::uint64_t>(max_candidates, 1000000)));
  }
  *capped = SweepPairs(scene.vertices, scene.mesh0.faces, max_candidates, [&](int vertex_slot, int face_index) {
    const VertexProxy& vertex = scene.vertices[vertex_slot];
    const Face& face = scene.mesh0.faces[face_index];
    if (!ObjectPairAllowed(scene, vertex.object_id, face.object_id)) {
      return false;
    }
    if (vertex.object_id == face.object_id && FaceContainsVertex(face, vertex.id)) {
      return false;
    }
    candidates.push_back(Candidate{vertex.id, face_index});
    return true;
  });
  return candidates;
}

std::vector<Candidate> BuildEECandidates(const PreparedScene& scene,
                                         std::uint64_t max_candidates,
                                         bool* capped) {
  std::vector<Candidate> candidates;
  if (max_candidates != 0) {
    candidates.reserve(static_cast<std::size_t>(std::min<std::uint64_t>(max_candidates, 1000000)));
  }
  *capped = SweepPairs(scene.edges, scene.edges, max_candidates, [&](int edge_a, int edge_b) {
    if (edge_a >= edge_b) {
      return false;
    }
    const Edge& a = scene.edges[edge_a];
    const Edge& b = scene.edges[edge_b];
    if (!ObjectPairAllowed(scene, a.object_id, b.object_id)) {
      return false;
    }
    if (a.object_id == b.object_id && EdgesShareVertex(a, b)) {
      return false;
    }
    candidates.push_back(Candidate{edge_a, edge_b});
    return true;
  });
  return candidates;
}

ticcd::Vector3 ToTi(const Vec3& v) {
  return ticcd::Vector3(static_cast<ticcd::Scalar>(v.x),
                        static_cast<ticcd::Scalar>(v.y),
                        static_cast<ticcd::Scalar>(v.z));
}

bool RunVF(const PreparedScene& scene, const Candidate& candidate, const Args& args) {
  const int vertex = candidate.a;
  const Face& face = scene.mesh0.faces[candidate.b];
  const ticcd::Array3 err(-1, -1, -1);
  ticcd::Scalar toi = std::numeric_limits<ticcd::Scalar>::infinity();
  ticcd::Scalar output_tolerance = static_cast<ticcd::Scalar>(args.tolerance);
  return ticcd::vertexFaceCCD(
      ToTi(scene.mesh0.vertices[vertex]),
      ToTi(scene.mesh0.vertices[face.v[0]]),
      ToTi(scene.mesh0.vertices[face.v[1]]),
      ToTi(scene.mesh0.vertices[face.v[2]]),
      ToTi(scene.mesh1.vertices[vertex]),
      ToTi(scene.mesh1.vertices[face.v[0]]),
      ToTi(scene.mesh1.vertices[face.v[1]]),
      ToTi(scene.mesh1.vertices[face.v[2]]),
      err,
      static_cast<ticcd::Scalar>(args.ms),
      toi,
      static_cast<ticcd::Scalar>(args.tolerance),
      static_cast<ticcd::Scalar>(args.t_max),
      args.max_itr,
      output_tolerance,
      false,
      ticcd::CCDRootFindingMethod::BREADTH_FIRST_SEARCH);
}

bool RunEE(const PreparedScene& scene, const Candidate& candidate, const Args& args) {
  const Edge& a = scene.edges[candidate.a];
  const Edge& b = scene.edges[candidate.b];
  const ticcd::Array3 err(-1, -1, -1);
  ticcd::Scalar toi = std::numeric_limits<ticcd::Scalar>::infinity();
  ticcd::Scalar output_tolerance = static_cast<ticcd::Scalar>(args.tolerance);
  return ticcd::edgeEdgeCCD(
      ToTi(scene.mesh0.vertices[a.a]),
      ToTi(scene.mesh0.vertices[a.b]),
      ToTi(scene.mesh0.vertices[b.a]),
      ToTi(scene.mesh0.vertices[b.b]),
      ToTi(scene.mesh1.vertices[a.a]),
      ToTi(scene.mesh1.vertices[a.b]),
      ToTi(scene.mesh1.vertices[b.a]),
      ToTi(scene.mesh1.vertices[b.b]),
      err,
      static_cast<ticcd::Scalar>(args.ms),
      toi,
      static_cast<ticcd::Scalar>(args.tolerance),
      static_cast<ticcd::Scalar>(args.t_max),
      args.max_itr,
      output_tolerance,
      false,
      ticcd::CCDRootFindingMethod::BREADTH_FIRST_SEARCH);
}

std::uint64_t GroupKey(const PreparedScene& scene, const std::string& kind, const Candidate& candidate) {
  int object_a = 0;
  int object_b = 0;
  if (kind == "vf") {
    object_a = scene.vertices[candidate.a].object_id;
    object_b = scene.mesh0.faces[candidate.b].object_id;
  } else {
    object_a = scene.edges[candidate.a].object_id;
    object_b = scene.edges[candidate.b].object_id;
  }
  if (object_b < object_a) {
    std::swap(object_a, object_b);
  }
  return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(object_a)) << 32) |
         static_cast<std::uint32_t>(object_b);
}

std::uint64_t Mix64(std::uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

double UnitHash(std::uint64_t x) {
  return static_cast<double>(Mix64(x) >> 11) * (1.0 / 9007199254740992.0);
}

Vec3 Center(const Aabb& box) {
  return Scale(Add(box.mn, box.mx), 0.5);
}

double Diagonal(const Aabb& box) {
  return Norm(Sub(box.mx, box.mn));
}

double DiagonalSquared(const Aabb& box) {
  const Vec3 extent = Sub(box.mx, box.mn);
  return Dot(extent, extent);
}

double DistanceSquared(const Vec3& a, const Vec3& b) {
  const Vec3 d = Sub(a, b);
  return Dot(d, d);
}

double OverlapVolume(const Aabb& a, const Aabb& b) {
  const double x = std::max(0.0, std::min(a.mx.x, b.mx.x) - std::max(a.mn.x, b.mn.x));
  const double y = std::max(0.0, std::min(a.mx.y, b.mx.y) - std::max(a.mn.y, b.mn.y));
  const double z = std::max(0.0, std::min(a.mx.z, b.mx.z) - std::max(a.mn.z, b.mn.z));
  return x * y * z;
}

Vec3 MeanPoint(const std::vector<Vec3>& vertices, const int* ids, int count) {
  Vec3 out{};
  for (int i = 0; i < count; ++i) {
    out = Add(out, vertices[ids[i]]);
  }
  return Scale(out, 1.0 / static_cast<double>(count));
}

FastSignals FastSignalsFromCandidate(const PreparedScene& scene,
                                     const std::string& kind,
                                     const Candidate& candidate) {
  Aabb a_box;
  Aabb b_box;
  if (kind == "vf") {
    const VertexProxy& vertex = scene.vertices[candidate.a];
    const Face& face = scene.mesh0.faces[candidate.b];
    a_box = vertex.box;
    b_box = face.box;
  } else {
    const Edge& edge_a = scene.edges[candidate.a];
    const Edge& edge_b = scene.edges[candidate.b];
    a_box = edge_a.box;
    b_box = edge_b.box;
  }

  const double diag2_a = DiagonalSquared(a_box);
  const double diag2_b = DiagonalSquared(b_box);
  const double diag2 = diag2_a + diag2_b;
  const double center_gap2 = DistanceSquared(Center(a_box), Center(b_box));
  const double overlap = OverlapVolume(a_box, b_box);
  FastSignals signals;
  signals.inv_gap = 1.0 / (1.0 + center_gap2);
  signals.rel_motion = 1.0 / (1.0 + std::abs(diag2_a - diag2_b));
  signals.overlap = std::log1p(overlap);
  signals.compactness = diag2 / (1.0e-9 + diag2 + center_gap2);
  return signals;
}

double FastLearnedFrontierScore(const PreparedScene& scene,
                               const std::string& kind,
                               const Candidate& candidate,
                               std::uint64_t index) {
  const FastSignals signals = FastSignalsFromCandidate(scene, kind, candidate);
  const double jitter = 1.0e-6 * UnitHash(index ^ (GroupKey(scene, kind, candidate) << 1));
  return 1.75 * signals.inv_gap + 0.85 * signals.compactness +
         0.35 * signals.rel_motion + 0.10 * signals.overlap + jitter;
}

double FastEnvelopeFrontierScore(const PreparedScene& scene,
                                 const std::string& kind,
                                 const Candidate& candidate,
                                 std::uint64_t index) {
  return FastLearnedFrontierScore(scene, kind, candidate, index);
}

std::array<Vec3, 8> CandidateRows(const PreparedScene& scene, const std::string& kind, const Candidate& candidate) {
  if (kind == "vf") {
    const int vertex = candidate.a;
    const Face& face = scene.mesh0.faces[candidate.b];
    return {
        scene.mesh0.vertices[vertex],
        scene.mesh0.vertices[face.v[0]],
        scene.mesh0.vertices[face.v[1]],
        scene.mesh0.vertices[face.v[2]],
        scene.mesh1.vertices[vertex],
        scene.mesh1.vertices[face.v[0]],
        scene.mesh1.vertices[face.v[1]],
        scene.mesh1.vertices[face.v[2]],
    };
  }
  const Edge& a = scene.edges[candidate.a];
  const Edge& b = scene.edges[candidate.b];
  return {
      scene.mesh0.vertices[a.a],
      scene.mesh0.vertices[a.b],
      scene.mesh0.vertices[b.a],
      scene.mesh0.vertices[b.b],
      scene.mesh1.vertices[a.a],
      scene.mesh1.vertices[a.b],
      scene.mesh1.vertices[b.a],
      scene.mesh1.vertices[b.b],
  };
}

std::array<double, 32> FeatureFromCandidate(const PreparedScene& scene,
                                            const std::string& kind,
                                            const Candidate& candidate) {
  const std::array<Vec3, 8> rows = CandidateRows(scene, kind, candidate);
  Aabb swept;
  for (const Vec3& p : rows) {
    Extend(&swept, p);
  }
  const Vec3 extent = Sub(swept.mx, swept.mn);
  std::array<Vec3, 4> pts0{rows[0], rows[1], rows[2], rows[3]};
  std::array<Vec3, 4> pts1{rows[4], rows[5], rows[6], rows[7]};
  std::array<Vec3, 4> motion{};
  for (int i = 0; i < 4; ++i) {
    motion[i] = Sub(pts1[i], pts0[i]);
  }
  const Vec3 mean_motion_a = Scale(Add(motion[0], motion[1]), 0.5);
  const Vec3 mean_motion_b = Scale(Add(motion[2], motion[3]), 0.5);
  const double rel = Norm(Sub(mean_motion_a, mean_motion_b));
  double speed = 0.0;
  for (const Vec3& m : motion) {
    speed = std::max(speed, Norm(m));
  }
  auto min_pair_gap = [](const std::array<Vec3, 4>& pts) {
    double out = std::numeric_limits<double>::infinity();
    for (int i = 0; i < 4; ++i) {
      for (int j = i + 1; j < 4; ++j) {
        out = std::min(out, Norm(Sub(pts[i], pts[j])));
      }
    }
    return out;
  };
  const double gap0 = min_pair_gap(pts0);
  const double gap1 = min_pair_gap(pts1);
  Vec3 mean0{};
  Vec3 mean1{};
  Vec3 mean_all{};
  for (int i = 0; i < 4; ++i) {
    mean0 = Add(mean0, pts0[i]);
    mean1 = Add(mean1, pts1[i]);
  }
  mean0 = Scale(mean0, 0.25);
  mean1 = Scale(mean1, 0.25);
  for (const Vec3& p : rows) {
    mean_all = Add(mean_all, p);
  }
  mean_all = Scale(mean_all, 0.125);
  double variance = 0.0;
  for (const Vec3& p : rows) {
    const Vec3 d = Sub(p, mean_all);
    variance += Dot(d, d);
  }
  variance /= static_cast<double>(rows.size());

  std::array<double, 32> f{};
  f[0] = kind == "vf" ? 1.0 : 0.0;
  f[1] = kind == "ee" ? 1.0 : 0.0;
  f[2] = extent.x;
  f[3] = extent.y;
  f[4] = extent.z;
  f[5] = Norm(extent);
  f[6] = std::log1p(std::max(extent.x, 1.0e-9) * std::max(extent.y, 1.0e-9) *
                    std::max(extent.z, 1.0e-9));
  f[7] = rel;
  f[8] = speed;
  f[9] = gap0;
  f[10] = gap1;
  f[11] = std::min(gap0, gap1);
  f[12] = mean_all.x;
  f[13] = mean_all.y;
  f[14] = mean_all.z;
  f[15] = Norm(Sub(mean1, mean0));
  f[16] = 0.0;
  for (const Vec3& p : rows) {
    f[16] = std::max(f[16], std::max({std::abs(p.x), std::abs(p.y), std::abs(p.z)}));
  }
  f[17] = std::sqrt(variance);
  f[18] = Norm(motion[0]);
  f[19] = Norm(motion[1]);
  f[20] = Norm(motion[2]);
  f[21] = Norm(motion[3]);
  f[22] = Norm(Sub(pts0[1], pts0[0]));
  f[23] = Norm(Sub(pts0[2], pts0[0]));
  f[24] = Norm(Sub(pts0[3], pts0[0]));
  f[25] = 1.0 / (1.0 + std::min(gap0, gap1));
  f[26] = f[8] / (1.0 + f[5]);
  f[27] = Dot(motion[0], motion[3]);
  f[28] = Norm(Cross(motion[0], motion[3]));
  f[29] = Norm(Sub(mean0, mean1));
  f[30] = 0.0;
  f[31] = 1.0;
  return f;
}

std::array<double, 32> FeatureFromCandidateWithIndex(const PreparedScene& scene,
                                                     const std::string& kind,
                                                     const Candidate& candidate,
                                                     std::uint64_t index) {
  std::array<double, 32> feature = FeatureFromCandidate(scene, kind, candidate);
  feature[30] = std::log1p(static_cast<double>(index));
  return feature;
}

TinyStpfWeights LoadTinyStpfWeights(const fs::path& path) {
  TinyStpfWeights weights;
  if (path.empty()) {
    return weights;
  }
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("failed to open STPF weights " + path.string());
  }
  auto expect = [&](const std::string& expected) {
    std::string tag;
    in >> tag;
    if (tag != expected) {
      throw std::runtime_error("bad STPF weight tag: expected " + expected + ", got " + tag);
    }
  };
  int input_dim = 0;
  int hidden0 = 0;
  int hidden1 = 0;
  expect("version");
  int version = 0;
  in >> version;
  if (version != 1) {
    throw std::runtime_error("unsupported STPF weight version");
  }
  expect("input_dim");
  in >> input_dim;
  expect("hidden0");
  in >> hidden0;
  expect("hidden1");
  in >> hidden1;
  if (input_dim != 32 || hidden0 != 64 || hidden1 != 32) {
    throw std::runtime_error("unexpected TinySTPF dimensions");
  }
  expect("mean");
  for (double& value : weights.mean) {
    in >> value;
  }
  expect("std");
  for (double& value : weights.std) {
    in >> value;
  }
  weights.w0.resize(64 * 32);
  weights.b0.resize(64);
  weights.w1.resize(32 * 64);
  weights.b1.resize(32);
  weights.w2.resize(32);
  expect("w0");
  for (double& value : weights.w0) {
    in >> value;
  }
  expect("b0");
  for (double& value : weights.b0) {
    in >> value;
  }
  expect("w1");
  for (double& value : weights.w1) {
    in >> value;
  }
  expect("b1");
  for (double& value : weights.b1) {
    in >> value;
  }
  expect("w2");
  for (double& value : weights.w2) {
    in >> value;
  }
  expect("b2");
  in >> weights.b2;
  weights.loaded = true;
  return weights;
}

double PredictTinyStpf(const TinyStpfWeights& weights, const std::array<double, 32>& feature) {
  std::array<double, 64> h0{};
  std::array<double, 32> h1{};
  for (int o = 0; o < 64; ++o) {
    double sum = weights.b0[o];
    for (int i = 0; i < 32; ++i) {
      const double x = (feature[i] - weights.mean[i]) / std::max(weights.std[i], 1.0e-12);
      sum += weights.w0[o * 32 + i] * x;
    }
    h0[o] = std::max(0.0, sum);
  }
  for (int o = 0; o < 32; ++o) {
    double sum = weights.b1[o];
    for (int i = 0; i < 64; ++i) {
      sum += weights.w1[o * 64 + i] * h0[i];
    }
    h1[o] = std::max(0.0, sum);
  }
  double out = weights.b2;
  for (int i = 0; i < 32; ++i) {
    out += weights.w2[i] * h1[i];
  }
  return out;
}

double Sigmoid(double value) {
  if (value >= 0.0) {
    const double z = std::exp(-value);
    return 1.0 / (1.0 + z);
  }
  const double z = std::exp(value);
  return z / (1.0 + z);
}

double ScoreCandidate(const PreparedScene& scene,
                      const std::string& kind,
                      const Candidate& candidate,
                      std::uint64_t index,
                      const std::string& policy,
                      const TinyStpfWeights& weights) {
  if (policy == "random") {
    return UnitHash(index ^ (kind == "vf" ? 0x767666ULL : 0x656565ULL) ^ (GroupKey(scene, kind, candidate) << 1));
  }
  const std::array<double, 32> feature = FeatureFromCandidateWithIndex(scene, kind, candidate, index);
  if (policy == "learned") {
    if (!weights.loaded) {
      throw std::runtime_error("learned policy requested without STPF weights");
    }
    return PredictTinyStpf(weights, feature);
  }
  if (policy == "learned_residual") {
    if (!weights.loaded) {
      throw std::runtime_error("learned residual policy requested without STPF weights");
    }
    const double learned = Sigmoid(PredictTinyStpf(weights, feature));
    return feature[25] + 0.08 * learned + 0.004 * feature[26] - 0.00025 * feature[30];
  }
  if (policy == "proximity") {
    return feature[25];
  }
  if (policy == "motion") {
    return feature[8];
  }
  throw std::runtime_error("unknown score policy: " + policy);
}

bool RunCandidate(const PreparedScene& scene, const Args& args, const std::string& kind, const Candidate& candidate) {
  return kind == "vf" ? RunVF(scene, candidate, args) : RunEE(scene, candidate, args);
}

std::string JsonEscape(const std::string& value);

bool ShouldExportFeatureLabel(const PreparedScene& scene,
                              const Args& args,
                              const std::string& kind,
                              const Candidate& candidate,
                              std::uint64_t index,
                              bool hit) {
  if (args.feature_jsonl.empty()) {
    return false;
  }
  if (hit) {
    return true;
  }
  if (args.feature_negative_stride == 0) {
    return false;
  }
  const std::uint64_t key = GroupKey(scene, kind, candidate);
  const std::uint64_t salt = kind == "vf" ? 0x5f76665f73616d70ULL : 0x5f65655f73616d70ULL;
  return (Mix64(index ^ (key * 0x9e3779b97f4a7c15ULL) ^ salt) % args.feature_negative_stride) == 0;
}

void WriteFeatureLabel(std::ostream& out,
                       const Args& args,
                       const PreparedScene& scene,
                       const std::string& kind,
                       const Candidate& candidate,
                       std::uint64_t index,
                       bool hit) {
  const std::uint64_t key = GroupKey(scene, kind, candidate);
  const std::array<double, 32> feature = FeatureFromCandidateWithIndex(scene, kind, candidate, index);
  out << std::setprecision(17)
      << "{\"scene\":\"" << JsonEscape(args.scene) << "\","
      << "\"kind\":\"" << JsonEscape(kind) << "\","
      << "\"group_key\":" << key << ","
      << "\"candidate_index\":" << index << ","
      << "\"label\":" << (hit ? 1 : 0) << ","
      << "\"features\":[";
  for (std::size_t i = 0; i < feature.size(); ++i) {
    if (i != 0) {
      out << ",";
    }
    out << feature[i];
  }
  out << "]}\n";
}

void FinalizeConfusion(MethodRow* row, const std::unordered_map<std::uint64_t, GroupTruth>& truth_by_group,
                       const std::unordered_map<std::uint64_t, bool>& detected_by_group) {
  row->group_count = truth_by_group.size();
  for (const auto& item : truth_by_group) {
    const bool truth = item.second.positive;
    const bool detected = detected_by_group.count(item.first) != 0 && detected_by_group.at(item.first);
    if (truth) {
      ++row->positive_groups;
      if (detected) {
        ++row->tp;
      } else {
        ++row->fn;
      }
    } else {
      ++row->negative_groups;
      if (detected) {
        ++row->fp;
      } else {
        ++row->tn;
      }
    }
  }
}

void CountScheduledExactCall(MethodRow* row,
                             const std::unordered_map<std::uint64_t, GroupTruth>& truth_by_group,
                             std::uint64_t key,
                             bool proposal_phase) {
  ++row->exact_calls;
  if (proposal_phase) {
    ++row->proposal_exact_calls;
  } else {
    ++row->fallback_exact_calls;
  }

  const auto truth = truth_by_group.find(key);
  const bool positive = truth != truth_by_group.end() && truth->second.positive;
  if (positive) {
    ++row->positive_exact_calls;
    if (proposal_phase) {
      ++row->positive_proposal_exact_calls;
    } else {
      ++row->positive_fallback_exact_calls;
    }
  } else {
    ++row->negative_exact_calls;
    if (proposal_phase) {
      ++row->negative_proposal_exact_calls;
    } else {
      ++row->negative_fallback_exact_calls;
    }
  }
}

MethodRow RunAllExactRows(const PreparedScene& scene,
                          const Args& args,
                          const std::string& kind,
                          const std::vector<Candidate>& candidates,
                          bool capped,
                          double envelope_ms,
                          std::unordered_map<std::uint64_t, GroupTruth>* truth_by_group,
                          std::ostream* feature_out = nullptr) {
  MethodRow row;
  row.method = "EnvelopeAllExact+TI";
  row.kind = kind;
  row.candidates = candidates.size();
  row.capped = capped;
  row.envelope_ms = envelope_ms;

  const auto total_begin = std::chrono::steady_clock::now();
  const auto exact_begin = total_begin;
  std::unordered_map<std::uint64_t, bool> detected_by_group;
  for (std::uint64_t index = 0; index < candidates.size(); ++index) {
    const Candidate& candidate = candidates[static_cast<std::size_t>(index)];
    const std::uint64_t key = GroupKey(scene, kind, candidate);
    (*truth_by_group)[key];
    const bool hit = RunCandidate(scene, args, kind, candidate);
    if (feature_out != nullptr && ShouldExportFeatureLabel(scene, args, kind, candidate, index, hit)) {
      WriteFeatureLabel(*feature_out, args, scene, kind, candidate, index, hit);
    }
    ++row.exact_calls;
    if (hit) {
      ++row.hit_count;
      (*truth_by_group)[key].positive = true;
      detected_by_group[key] = true;
      if (!row.detected_hit) {
        row.detected_hit = true;
        row.first_hit_rank = row.exact_calls;
      }
    }
  }
  const auto exact_end = std::chrono::steady_clock::now();
  row.exact_ms = std::chrono::duration<double, std::milli>(exact_end - exact_begin).count();
  row.total_ms = row.envelope_ms + std::chrono::duration<double, std::milli>(exact_end - total_begin).count();
  FinalizeConfusion(&row, *truth_by_group, detected_by_group);
  return row;
}

MethodRow RunScheduledAnyHitRows(const PreparedScene& scene,
                                 const Args& args,
                                 const std::string& kind,
                                 const std::vector<Candidate>& candidates,
                                 bool capped,
                                 double envelope_ms,
                                 const std::string& method_name,
                                 const std::string& score_policy,
                                 const TinyStpfWeights& weights,
                                 const std::unordered_map<std::uint64_t, GroupTruth>& truth_by_group) {
  MethodRow row;
  row.method = method_name;
  row.kind = kind;
  row.candidates = candidates.size();
  row.capped = capped;
  row.envelope_ms = envelope_ms;
  row.effective_proposal_k = args.proposal_top_k;

  const auto total_begin = std::chrono::steady_clock::now();
  const auto order_begin = total_begin;
  struct MinScore {
    bool operator()(const ScoredCandidate& a, const ScoredCandidate& b) const {
      return a.score > b.score;
    }
  } min_score;
  std::unordered_map<std::uint64_t, std::vector<ScoredCandidate>> top_by_group;
  top_by_group.reserve(truth_by_group.size() * 2 + 1);
  for (std::uint64_t index = 0; index < candidates.size(); ++index) {
    const Candidate& candidate = candidates[static_cast<std::size_t>(index)];
    const std::uint64_t key = GroupKey(scene, kind, candidate);
    auto& heap = top_by_group[key];
    const double score = ScoreCandidate(scene, kind, candidate, index, score_policy, weights);
    if (heap.size() < args.proposal_top_k) {
      heap.push_back(ScoredCandidate{score, index});
      std::push_heap(heap.begin(), heap.end(), min_score);
    } else if (!heap.empty() && score > heap.front().score) {
      std::pop_heap(heap.begin(), heap.end(), min_score);
      heap.back() = ScoredCandidate{score, index};
      std::push_heap(heap.begin(), heap.end(), min_score);
    }
  }
  for (auto& item : top_by_group) {
    std::sort(item.second.begin(), item.second.end(), [](const ScoredCandidate& a, const ScoredCandidate& b) {
      return a.score > b.score;
    });
  }
  const auto order_end = std::chrono::steady_clock::now();
  row.ordering_ms = std::chrono::duration<double, std::milli>(order_end - order_begin).count();

  std::unordered_map<std::uint64_t, bool> detected_by_group;
  std::unordered_map<std::uint64_t, std::uint64_t> proposal_tested_by_group;
  std::unordered_map<std::uint64_t, std::uint64_t> fallback_tested_by_group;
  std::unordered_set<std::uint64_t> top_indices;
  std::unordered_set<std::uint64_t> fallback_groups;
  proposal_tested_by_group.reserve(truth_by_group.size() * 2 + 1);
  fallback_tested_by_group.reserve(truth_by_group.size() * 2 + 1);
  top_indices.reserve(truth_by_group.size() * std::min<std::size_t>(args.proposal_top_k, 4096) + 1);

  const auto exact_begin = std::chrono::steady_clock::now();
  for (const auto& item : truth_by_group) {
    const std::uint64_t key = item.first;
    const auto found = top_by_group.find(key);
    if (found == top_by_group.end()) {
      fallback_groups.insert(key);
      continue;
    }
    for (const ScoredCandidate& scored : found->second) {
      top_indices.insert(scored.index);
      const Candidate& candidate = candidates[static_cast<std::size_t>(scored.index)];
      const bool hit = RunCandidate(scene, args, kind, candidate);
      CountScheduledExactCall(&row, truth_by_group, key, true);
      ++proposal_tested_by_group[key];
      if (hit) {
        ++row.hit_count;
        detected_by_group[key] = true;
        if (item.second.positive) {
          ++row.positive_proposal_hits;
          ++row.positive_first_hit_groups;
          row.positive_first_hit_rank_sum += proposal_tested_by_group[key];
        }
        if (!row.detected_hit) {
          row.detected_hit = true;
          row.first_hit_rank = row.exact_calls;
        }
        break;
      }
    }
    if (detected_by_group.count(key) == 0) {
      fallback_groups.insert(key);
    }
  }

  const std::unordered_set<std::uint64_t> fallback_attempted = fallback_groups;
  for (std::uint64_t index = 0; index < candidates.size() && !fallback_groups.empty(); ++index) {
    if (top_indices.count(index) != 0) {
      continue;
    }
    const Candidate& candidate = candidates[static_cast<std::size_t>(index)];
    const std::uint64_t key = GroupKey(scene, kind, candidate);
    if (fallback_groups.count(key) == 0) {
      continue;
    }
    const bool hit = RunCandidate(scene, args, kind, candidate);
    CountScheduledExactCall(&row, truth_by_group, key, false);
    ++fallback_tested_by_group[key];
    if (hit) {
      ++row.hit_count;
      detected_by_group[key] = true;
      fallback_groups.erase(key);
      const auto truth = truth_by_group.find(key);
      if (truth != truth_by_group.end() && truth->second.positive) {
        ++row.positive_fallback_hits;
        ++row.positive_first_hit_groups;
        row.positive_first_hit_rank_sum += proposal_tested_by_group[key] + fallback_tested_by_group[key];
      }
      if (!row.detected_hit) {
        row.detected_hit = true;
        row.first_hit_rank = row.exact_calls;
      }
    }
  }
  row.fallback_groups = fallback_attempted.size();
  const auto exact_end = std::chrono::steady_clock::now();
  row.exact_ms = std::chrono::duration<double, std::milli>(exact_end - exact_begin).count();
  row.total_ms = row.envelope_ms + row.ordering_ms + row.exact_ms;
  FinalizeConfusion(&row, truth_by_group, detected_by_group);
  return row;
}

MethodRow RunOptimizedLearnedAnyHitRows(const PreparedScene& scene,
                                        const Args& args,
                                        const std::string& kind,
                                        const std::vector<Candidate>& candidates,
                                        bool capped,
                                        double envelope_ms,
                                        const TinyStpfWeights& weights,
                                        const std::unordered_map<std::uint64_t, GroupTruth>& truth_by_group) {
  if (args.optimized_random_gate_object_count > 0 &&
      scene.object_count >= args.optimized_random_gate_object_count) {
    MethodRow row = RunScheduledAnyHitRows(scene, args, kind, candidates, capped, envelope_ms,
                                           "OptimizedFrozenLearnedAnyHit+TI", "random", weights, truth_by_group);
    row.random_gate_used = true;
    return row;
  }
  if (!weights.loaded) {
    throw std::runtime_error("optimized learned policy requested without STPF weights");
  }
  MethodRow row;
  row.method = "OptimizedFrozenLearnedAnyHit+TI";
  row.kind = kind;
  row.candidates = candidates.size();
  row.capped = capped;
  row.envelope_ms = envelope_ms;

  const auto total_begin = std::chrono::steady_clock::now();
  const auto order_begin = total_begin;
  const std::size_t frontier_k = std::max<std::size_t>(1, args.optimized_frontier_k);
  const std::size_t proposal_k = std::max<std::size_t>(1, std::min(args.proposal_top_k, frontier_k));
  row.effective_frontier_k = frontier_k;
  row.effective_proposal_k = proposal_k;

  struct MinScore {
    bool operator()(const ScoredCandidate& a, const ScoredCandidate& b) const {
      return a.score > b.score;
    }
  } min_score;

  std::unordered_map<std::uint64_t, std::vector<ScoredCandidate>> frontier_by_group;
  frontier_by_group.reserve(truth_by_group.size() * 2 + 1);
  std::unordered_map<std::uint64_t, std::uint64_t> scanned_by_group;
  scanned_by_group.reserve(truth_by_group.size() * 2 + 1);
  for (std::uint64_t index = 0; index < candidates.size(); ++index) {
    const Candidate& candidate = candidates[static_cast<std::size_t>(index)];
    const std::uint64_t key = GroupKey(scene, kind, candidate);
    std::uint64_t& scanned = scanned_by_group[key];
    ++scanned;
    if (args.optimized_scan_limit_per_group != 0 && scanned > args.optimized_scan_limit_per_group) {
      continue;
    }
    auto& heap = frontier_by_group[key];
    const double score = FastLearnedFrontierScore(scene, kind, candidate, index);
    if (heap.size() < frontier_k) {
      heap.push_back(ScoredCandidate{score, index});
      std::push_heap(heap.begin(), heap.end(), min_score);
    } else if (!heap.empty() && score > heap.front().score) {
      std::pop_heap(heap.begin(), heap.end(), min_score);
      heap.back() = ScoredCandidate{score, index};
      std::push_heap(heap.begin(), heap.end(), min_score);
    }
  }

  std::unordered_map<std::uint64_t, std::vector<ScoredCandidate>> top_by_group;
  top_by_group.reserve(truth_by_group.size() * 2 + 1);
  for (auto& item : frontier_by_group) {
    std::vector<LearnedFrontierCandidate> learned;
    learned.reserve(item.second.size());
    double sum_learned = 0.0;
    double sum_proximity = 0.0;
    double sum_motion = 0.0;
    double sum_density = 0.0;
    for (const ScoredCandidate& scored : item.second) {
      const Candidate& candidate = candidates[static_cast<std::size_t>(scored.index)];
      const std::array<double, 32> feature = FeatureFromCandidateWithIndex(scene, kind, candidate, scored.index);
      LearnedFrontierCandidate out;
      out.index = scored.index;
      out.learned = PredictTinyStpf(weights, feature);
      out.proximity = feature[25];
      out.motion = feature[8];
      out.density = feature[26];
      learned.push_back(out);
      sum_learned += out.learned;
      sum_proximity += out.proximity;
      sum_motion += out.motion;
      sum_density += out.density;
    }
    const double n = std::max<double>(1.0, static_cast<double>(learned.size()));
    const double mean_learned = sum_learned / n;
    const double mean_proximity = sum_proximity / n;
    const double mean_motion = sum_motion / n;
    const double mean_density = sum_density / n;
    double var_learned = 0.0;
    double var_proximity = 0.0;
    double var_motion = 0.0;
    double var_density = 0.0;
    for (const LearnedFrontierCandidate& candidate : learned) {
      var_learned += (candidate.learned - mean_learned) * (candidate.learned - mean_learned);
      var_proximity += (candidate.proximity - mean_proximity) * (candidate.proximity - mean_proximity);
      var_motion += (candidate.motion - mean_motion) * (candidate.motion - mean_motion);
      var_density += (candidate.density - mean_density) * (candidate.density - mean_density);
    }
    const double std_learned = std::sqrt(var_learned / n) + 1.0e-12;
    const double std_proximity = std::sqrt(var_proximity / n) + 1.0e-12;
    const double std_motion = std::sqrt(var_motion / n) + 1.0e-12;
    const double std_density = std::sqrt(var_density / n) + 1.0e-12;
    for (LearnedFrontierCandidate& candidate : learned) {
      const double z_learned = (candidate.learned - mean_learned) / std_learned;
      const double z_proximity = (candidate.proximity - mean_proximity) / std_proximity;
      const double z_motion = (candidate.motion - mean_motion) / std_motion;
      const double z_density = (candidate.density - mean_density) / std_density;
      // A fixed global learned+geometry blend avoids per-scene oracle tuning while
      // keeping the frozen STPF score in the final ranking decision.
      candidate.score = 1.20 * z_proximity + 0.15 * z_learned + 0.05 * z_motion + 0.05 * z_density;
    }
    std::sort(learned.begin(), learned.end(), [](const LearnedFrontierCandidate& a,
                                                  const LearnedFrontierCandidate& b) {
      return a.score > b.score;
    });
    auto& top = top_by_group[item.first];
    const std::size_t keep = std::min<std::size_t>(proposal_k, learned.size());
    top.reserve(keep);
    for (std::size_t i = 0; i < keep; ++i) {
      top.push_back(ScoredCandidate{learned[i].score, learned[i].index});
    }
  }
  const auto order_end = std::chrono::steady_clock::now();
  row.ordering_ms = std::chrono::duration<double, std::milli>(order_end - order_begin).count();

  std::unordered_map<std::uint64_t, bool> detected_by_group;
  std::unordered_map<std::uint64_t, std::uint64_t> proposal_tested_by_group;
  std::unordered_map<std::uint64_t, std::uint64_t> fallback_tested_by_group;
  std::unordered_set<std::uint64_t> top_indices;
  std::unordered_set<std::uint64_t> fallback_groups;
  proposal_tested_by_group.reserve(truth_by_group.size() * 2 + 1);
  fallback_tested_by_group.reserve(truth_by_group.size() * 2 + 1);
  top_indices.reserve(truth_by_group.size() * std::min<std::size_t>(proposal_k, 4096) + 1);

  const auto exact_begin = std::chrono::steady_clock::now();
  for (const auto& item : truth_by_group) {
    const std::uint64_t key = item.first;
    const auto found = top_by_group.find(key);
    if (found == top_by_group.end()) {
      fallback_groups.insert(key);
      continue;
    }
    for (const ScoredCandidate& scored : found->second) {
      top_indices.insert(scored.index);
      const Candidate& candidate = candidates[static_cast<std::size_t>(scored.index)];
      const bool hit = RunCandidate(scene, args, kind, candidate);
      CountScheduledExactCall(&row, truth_by_group, key, true);
      ++proposal_tested_by_group[key];
      if (hit) {
        ++row.hit_count;
        detected_by_group[key] = true;
        if (item.second.positive) {
          ++row.positive_proposal_hits;
          ++row.positive_first_hit_groups;
          row.positive_first_hit_rank_sum += proposal_tested_by_group[key];
        }
        if (!row.detected_hit) {
          row.detected_hit = true;
          row.first_hit_rank = row.exact_calls;
        }
        break;
      }
    }
    if (detected_by_group.count(key) == 0) {
      fallback_groups.insert(key);
    }
  }

  const std::unordered_set<std::uint64_t> fallback_attempted = fallback_groups;
  for (std::uint64_t index = 0; index < candidates.size() && !fallback_groups.empty(); ++index) {
    if (top_indices.count(index) != 0) {
      continue;
    }
    const Candidate& candidate = candidates[static_cast<std::size_t>(index)];
    const std::uint64_t key = GroupKey(scene, kind, candidate);
    if (fallback_groups.count(key) == 0) {
      continue;
    }
    const bool hit = RunCandidate(scene, args, kind, candidate);
    CountScheduledExactCall(&row, truth_by_group, key, false);
    ++fallback_tested_by_group[key];
    if (hit) {
      ++row.hit_count;
      detected_by_group[key] = true;
      fallback_groups.erase(key);
      const auto truth = truth_by_group.find(key);
      if (truth != truth_by_group.end() && truth->second.positive) {
        ++row.positive_fallback_hits;
        ++row.positive_first_hit_groups;
        row.positive_first_hit_rank_sum += proposal_tested_by_group[key] + fallback_tested_by_group[key];
      }
      if (!row.detected_hit) {
        row.detected_hit = true;
        row.first_hit_rank = row.exact_calls;
      }
    }
  }
  row.fallback_groups = fallback_attempted.size();
  const auto exact_end = std::chrono::steady_clock::now();
  row.exact_ms = std::chrono::duration<double, std::milli>(exact_end - exact_begin).count();
  row.total_ms = row.envelope_ms + row.ordering_ms + row.exact_ms;
  FinalizeConfusion(&row, truth_by_group, detected_by_group);
  return row;
}

double FairFrontierRankingScore(const PreparedScene& scene,
                                const std::string& kind,
                                const Candidate& candidate,
                                std::uint64_t index,
                                const std::string& ranking_policy,
                                const TinyStpfWeights& weights) {
  if (ranking_policy == "random") {
    return UnitHash(index ^ (kind == "vf" ? 0x9e3779b97f4a7c15ULL : 0xbf58476d1ce4e5b9ULL) ^
                    (GroupKey(scene, kind, candidate) << 1));
  }
  const std::array<double, 32> feature = FeatureFromCandidateWithIndex(scene, kind, candidate, index);
  if (ranking_policy == "learned") {
    if (!weights.loaded) {
      throw std::runtime_error("fair learned policy requested without STPF weights");
    }
    return PredictTinyStpf(weights, feature);
  }
  if (ranking_policy == "learned_residual") {
    if (!weights.loaded) {
      throw std::runtime_error("fair learned residual policy requested without STPF weights");
    }
    const double learned = Sigmoid(PredictTinyStpf(weights, feature));
    return feature[25] + 0.08 * learned + 0.004 * feature[26] - 0.00025 * feature[30];
  }
  if (ranking_policy == "proximity") {
    return feature[25];
  }
  if (ranking_policy == "motion") {
    return feature[8];
  }
  throw std::runtime_error("unknown fair frontier ranking policy: " + ranking_policy);
}

MethodRow RunFairFrontierAnyHitRows(const PreparedScene& scene,
                                    const Args& args,
                                    const std::string& kind,
                                    const std::vector<Candidate>& candidates,
                                    bool capped,
                                    double envelope_ms,
                                    const std::string& method_name,
                                    const std::string& ranking_policy,
                                    const TinyStpfWeights& weights,
                                    const std::unordered_map<std::uint64_t, GroupTruth>& truth_by_group) {
  MethodRow row;
  row.method = method_name;
  row.kind = kind;
  row.candidates = candidates.size();
  row.capped = capped;
  row.envelope_ms = envelope_ms;

  const auto total_begin = std::chrono::steady_clock::now();
  const auto order_begin = total_begin;
  const std::size_t frontier_k = std::max<std::size_t>(1, args.optimized_frontier_k);
  const std::size_t proposal_k = std::max<std::size_t>(1, std::min(args.proposal_top_k, frontier_k));
  row.effective_frontier_k = frontier_k;
  row.effective_proposal_k = proposal_k;

  struct MinScore {
    bool operator()(const ScoredCandidate& a, const ScoredCandidate& b) const {
      return a.score > b.score;
    }
  } min_score;

  std::unordered_map<std::uint64_t, std::vector<ScoredCandidate>> frontier_by_group;
  frontier_by_group.reserve(truth_by_group.size() * 2 + 1);
  std::unordered_map<std::uint64_t, std::uint64_t> scanned_by_group;
  scanned_by_group.reserve(truth_by_group.size() * 2 + 1);
  for (std::uint64_t index = 0; index < candidates.size(); ++index) {
    const Candidate& candidate = candidates[static_cast<std::size_t>(index)];
    const std::uint64_t key = GroupKey(scene, kind, candidate);
    std::uint64_t& scanned = scanned_by_group[key];
    ++scanned;
    if (args.optimized_scan_limit_per_group != 0 && scanned > args.optimized_scan_limit_per_group) {
      continue;
    }
    auto& heap = frontier_by_group[key];
    const double score = FastEnvelopeFrontierScore(scene, kind, candidate, index);
    if (heap.size() < frontier_k) {
      heap.push_back(ScoredCandidate{score, index});
      std::push_heap(heap.begin(), heap.end(), min_score);
    } else if (!heap.empty() && score > heap.front().score) {
      std::pop_heap(heap.begin(), heap.end(), min_score);
      heap.back() = ScoredCandidate{score, index};
      std::push_heap(heap.begin(), heap.end(), min_score);
    }
  }

  std::unordered_map<std::uint64_t, std::vector<ScoredCandidate>> top_by_group;
  top_by_group.reserve(truth_by_group.size() * 2 + 1);
  for (auto& item : frontier_by_group) {
    std::vector<ScoredCandidate> ranked;
    ranked.reserve(item.second.size());
    for (const ScoredCandidate& frontier_candidate : item.second) {
      const Candidate& candidate = candidates[static_cast<std::size_t>(frontier_candidate.index)];
      const double score = FairFrontierRankingScore(scene, kind, candidate, frontier_candidate.index,
                                                    ranking_policy, weights);
      ranked.push_back(ScoredCandidate{score, frontier_candidate.index});
    }
    if (ranked.size() > proposal_k) {
      std::nth_element(ranked.begin(), ranked.begin() + static_cast<std::ptrdiff_t>(proposal_k), ranked.end(),
                       [](const ScoredCandidate& a, const ScoredCandidate& b) {
                         return a.score > b.score;
                       });
      ranked.resize(proposal_k);
    }
    std::sort(ranked.begin(), ranked.end(), [](const ScoredCandidate& a, const ScoredCandidate& b) {
      return a.score > b.score;
    });
    top_by_group.emplace(item.first, std::move(ranked));
  }
  const auto order_end = std::chrono::steady_clock::now();
  row.ordering_ms = std::chrono::duration<double, std::milli>(order_end - order_begin).count();

  std::unordered_map<std::uint64_t, bool> detected_by_group;
  std::unordered_map<std::uint64_t, std::uint64_t> proposal_tested_by_group;
  std::unordered_map<std::uint64_t, std::uint64_t> fallback_tested_by_group;
  std::unordered_set<std::uint64_t> top_indices;
  std::unordered_set<std::uint64_t> fallback_groups;
  proposal_tested_by_group.reserve(truth_by_group.size() * 2 + 1);
  fallback_tested_by_group.reserve(truth_by_group.size() * 2 + 1);
  top_indices.reserve(truth_by_group.size() * std::min<std::size_t>(proposal_k, 4096) + 1);

  const auto exact_begin = std::chrono::steady_clock::now();
  for (const auto& item : truth_by_group) {
    const std::uint64_t key = item.first;
    const auto found = top_by_group.find(key);
    if (found == top_by_group.end()) {
      fallback_groups.insert(key);
      continue;
    }
    for (const ScoredCandidate& scored : found->second) {
      top_indices.insert(scored.index);
      const Candidate& candidate = candidates[static_cast<std::size_t>(scored.index)];
      const bool hit = RunCandidate(scene, args, kind, candidate);
      CountScheduledExactCall(&row, truth_by_group, key, true);
      ++proposal_tested_by_group[key];
      if (hit) {
        ++row.hit_count;
        detected_by_group[key] = true;
        if (item.second.positive) {
          ++row.positive_proposal_hits;
          ++row.positive_first_hit_groups;
          row.positive_first_hit_rank_sum += proposal_tested_by_group[key];
        }
        if (!row.detected_hit) {
          row.detected_hit = true;
          row.first_hit_rank = row.exact_calls;
        }
        break;
      }
    }
    if (detected_by_group.count(key) == 0) {
      fallback_groups.insert(key);
    }
  }

  const std::unordered_set<std::uint64_t> fallback_attempted = fallback_groups;
  for (std::uint64_t index = 0; index < candidates.size() && !fallback_groups.empty(); ++index) {
    if (top_indices.count(index) != 0) {
      continue;
    }
    const Candidate& candidate = candidates[static_cast<std::size_t>(index)];
    const std::uint64_t key = GroupKey(scene, kind, candidate);
    if (fallback_groups.count(key) == 0) {
      continue;
    }
    const bool hit = RunCandidate(scene, args, kind, candidate);
    CountScheduledExactCall(&row, truth_by_group, key, false);
    ++fallback_tested_by_group[key];
    if (hit) {
      ++row.hit_count;
      detected_by_group[key] = true;
      fallback_groups.erase(key);
      const auto truth = truth_by_group.find(key);
      if (truth != truth_by_group.end() && truth->second.positive) {
        ++row.positive_fallback_hits;
        ++row.positive_first_hit_groups;
        row.positive_first_hit_rank_sum += proposal_tested_by_group[key] + fallback_tested_by_group[key];
      }
      if (!row.detected_hit) {
        row.detected_hit = true;
        row.first_hit_rank = row.exact_calls;
      }
    }
  }
  row.fallback_groups = fallback_attempted.size();
  const auto exact_end = std::chrono::steady_clock::now();
  row.exact_ms = std::chrono::duration<double, std::milli>(exact_end - exact_begin).count();
  row.total_ms = row.envelope_ms + row.ordering_ms + row.exact_ms;
  FinalizeConfusion(&row, truth_by_group, detected_by_group);
  return row;
}

std::string JsonEscape(const std::string& value) {
  std::string out;
  out.reserve(value.size() + 8);
  for (const char ch : value) {
    if (ch == '"' || ch == '\\') {
      out.push_back('\\');
    }
    if (ch == '\n') {
      out += "\\n";
    } else if (ch == '\r') {
      out += "\\r";
    } else {
      out.push_back(ch);
    }
  }
  return out;
}

void WriteRow(std::ostream& out,
              const Args& args,
              const PreparedScene& scene,
              const MethodRow& row,
              double load_ms,
              double prepare_ms) {
  const double throughput = row.exact_ms > 0.0 ? 1000.0 * static_cast<double>(row.exact_calls) / row.exact_ms : 0.0;
  out << "{"
      << "\"scene\":\"" << JsonEscape(args.scene) << "\","
      << "\"kind\":\"" << JsonEscape(row.kind) << "\","
      << "\"method\":\"" << JsonEscape(row.method) << "\","
      << "\"frame0\":\"" << JsonEscape(args.frame0.string()) << "\","
      << "\"frame1\":\"" << JsonEscape(args.frame1.string()) << "\","
      << "\"vertices\":" << scene.mesh0.vertices.size() << ","
      << "\"faces\":" << scene.mesh0.faces.size() << ","
      << "\"edges\":" << scene.edges.size() << ","
      << "\"objects\":" << scene.object_count << ","
      << "\"object_envelopes\":" << scene.object_envelope_count << ","
      << "\"candidate_count\":" << row.candidates << ","
      << "\"candidate_capped\":" << (row.capped ? "true" : "false") << ","
      << "\"effective_frontier_k\":" << row.effective_frontier_k << ","
      << "\"effective_proposal_k\":" << row.effective_proposal_k << ","
      << "\"random_gate_used\":" << (row.random_gate_used ? "true" : "false") << ","
      << "\"group_count\":" << row.group_count << ","
      << "\"positive_groups\":" << row.positive_groups << ","
      << "\"negative_groups\":" << row.negative_groups << ","
      << "\"fallback_groups\":" << row.fallback_groups << ","
      << "\"fallback_rate\":"
      << (row.group_count > 0 ? static_cast<double>(row.fallback_groups) / static_cast<double>(row.group_count) : 0.0)
      << ","
      << "\"tp\":" << row.tp << ","
      << "\"tn\":" << row.tn << ","
      << "\"fp\":" << row.fp << ","
      << "\"fn\":" << row.fn << ","
      << "\"exact_calls\":" << row.exact_calls << ","
      << "\"proposal_exact_calls\":" << row.proposal_exact_calls << ","
      << "\"fallback_exact_calls\":" << row.fallback_exact_calls << ","
      << "\"positive_exact_calls\":" << row.positive_exact_calls << ","
      << "\"negative_exact_calls\":" << row.negative_exact_calls << ","
      << "\"positive_proposal_exact_calls\":" << row.positive_proposal_exact_calls << ","
      << "\"positive_fallback_exact_calls\":" << row.positive_fallback_exact_calls << ","
      << "\"negative_proposal_exact_calls\":" << row.negative_proposal_exact_calls << ","
      << "\"negative_fallback_exact_calls\":" << row.negative_fallback_exact_calls << ","
      << "\"positive_proposal_hits\":" << row.positive_proposal_hits << ","
      << "\"positive_fallback_hits\":" << row.positive_fallback_hits << ","
      << "\"positive_first_hit_rank_sum\":" << row.positive_first_hit_rank_sum << ","
      << "\"positive_first_hit_groups\":" << row.positive_first_hit_groups << ","
      << "\"native_hit_count\":" << row.hit_count << ","
      << "\"detected_hit\":" << (row.detected_hit ? "true" : "false") << ","
      << "\"first_hit_rank\":" << row.first_hit_rank << ","
      << "\"load_ms\":" << load_ms << ","
      << "\"prepare_ms\":" << prepare_ms << ","
      << "\"envelope_ms\":" << row.envelope_ms << ","
      << "\"ordering_ms\":" << row.ordering_ms << ","
      << "\"native_exact_backend_ms\":" << row.exact_ms << ","
      << "\"total_wall_ms\":" << (load_ms + prepare_ms + row.total_ms) << ","
      << "\"exact_calls_per_second\":" << throughput << ","
      << "\"proposal_top_k\":" << args.proposal_top_k << ","
      << "\"optimized_frontier_k\":" << args.optimized_frontier_k << ","
      << "\"optimized_scan_limit_per_group\":" << args.optimized_scan_limit_per_group << ","
      << "\"optimized_random_gate_object_count\":" << args.optimized_random_gate_object_count << ","
      << "\"thickness\":" << args.thickness << ","
      << "\"tolerance\":" << args.tolerance << ","
      << "\"max_itr\":" << args.max_itr
      << "}\n";
}

Args ParseArgs(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    auto need_value = [&](const std::string& name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error("missing value for " + name);
      }
      return argv[++i];
    };
    if (key == "--frame0") {
      args.frame0 = need_value(key);
    } else if (key == "--frame1") {
      args.frame1 = need_value(key);
    } else if (key == "--output-jsonl") {
      args.output_jsonl = need_value(key);
    } else if (key == "--scene") {
      args.scene = need_value(key);
    } else if (key == "--thickness") {
      args.thickness = std::stod(need_value(key));
    } else if (key == "--ms") {
      args.ms = std::stod(need_value(key));
    } else if (key == "--tolerance") {
      args.tolerance = std::stod(need_value(key));
    } else if (key == "--t-max") {
      args.t_max = std::stod(need_value(key));
    } else if (key == "--max-itr") {
      args.max_itr = std::stol(need_value(key));
    } else if (key == "--max-vf-candidates") {
      args.max_vf_candidates = static_cast<std::uint64_t>(std::stoull(need_value(key)));
    } else if (key == "--max-ee-candidates") {
      args.max_ee_candidates = static_cast<std::uint64_t>(std::stoull(need_value(key)));
    } else if (key == "--stpf-weights") {
      args.stpf_weights = need_value(key);
    } else if (key == "--proposal-top-k") {
      args.proposal_top_k = static_cast<std::size_t>(std::stoull(need_value(key)));
    } else if (key == "--optimized-frontier-k") {
      args.optimized_frontier_k = static_cast<std::size_t>(std::stoull(need_value(key)));
    } else if (key == "--optimized-scan-limit-per-group") {
      args.optimized_scan_limit_per_group = static_cast<std::uint64_t>(std::stoull(need_value(key)));
    } else if (key == "--optimized-random-gate-object-count") {
      args.optimized_random_gate_object_count = std::stoi(need_value(key));
    } else if (key == "--feature-jsonl") {
      args.feature_jsonl = need_value(key);
    } else if (key == "--feature-negative-stride") {
      args.feature_negative_stride = static_cast<std::uint64_t>(std::stoull(need_value(key)));
    } else if (key == "--feature-export-only") {
      args.feature_export_only = true;
    } else if (key == "--exclude-self-object-pairs") {
      args.exclude_self_object_pairs = true;
    } else {
      throw std::runtime_error("unknown argument: " + key);
    }
  }
  if (args.frame0.empty() || args.frame1.empty() || args.output_jsonl.empty()) {
    throw std::runtime_error("--frame0, --frame1 and --output-jsonl are required");
  }
  return args;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = ParseArgs(argc, argv);
    const auto load_begin = std::chrono::steady_clock::now();
    PreparedScene scene = PrepareScene(args.frame0, args.frame1, args.thickness, args.exclude_self_object_pairs);
    const auto load_end = std::chrono::steady_clock::now();
    const double load_prepare_ms = std::chrono::duration<double, std::milli>(load_end - load_begin).count();
    const TinyStpfWeights stpf_weights = LoadTinyStpfWeights(args.stpf_weights);

    fs::create_directories(args.output_jsonl.parent_path());
    std::ofstream out(args.output_jsonl);
    if (!out) {
      throw std::runtime_error("failed to open output " + args.output_jsonl.string());
    }
    std::ofstream feature_out;
    if (!args.feature_jsonl.empty()) {
      fs::create_directories(args.feature_jsonl.parent_path());
      feature_out.open(args.feature_jsonl);
      if (!feature_out) {
        throw std::runtime_error("failed to open feature export " + args.feature_jsonl.string());
      }
    }
    std::ostream* feature_stream = feature_out ? &feature_out : nullptr;

    const auto vf_begin = std::chrono::steady_clock::now();
    bool vf_capped = false;
    std::vector<Candidate> vf = BuildVFCandidates(scene, args.max_vf_candidates, &vf_capped);
    const auto vf_end = std::chrono::steady_clock::now();
    const double vf_envelope_ms = std::chrono::duration<double, std::milli>(vf_end - vf_begin).count();
    std::unordered_map<std::uint64_t, GroupTruth> vf_truth;
    WriteRow(out, args, scene, RunAllExactRows(scene, args, "vf", vf, vf_capped, vf_envelope_ms, &vf_truth, feature_stream), load_prepare_ms, 0.0);
    if (args.feature_export_only) {
      const auto ee_begin = std::chrono::steady_clock::now();
      bool ee_capped = false;
      std::vector<Candidate> ee = BuildEECandidates(scene, args.max_ee_candidates, &ee_capped);
      const auto ee_end = std::chrono::steady_clock::now();
      const double ee_envelope_ms = std::chrono::duration<double, std::milli>(ee_end - ee_begin).count();
      std::unordered_map<std::uint64_t, GroupTruth> ee_truth;
      WriteRow(out, args, scene, RunAllExactRows(scene, args, "ee", ee, ee_capped, ee_envelope_ms, &ee_truth, feature_stream), load_prepare_ms, 0.0);
      return 0;
    }
    WriteRow(out, args, scene,
             RunFairFrontierAnyHitRows(scene, args, "vf", vf, vf_capped, vf_envelope_ms,
                                        "FairFrontierLearnedAnyHit+TI", "learned", stpf_weights, vf_truth),
             load_prepare_ms, 0.0);
    WriteRow(out, args, scene,
             RunFairFrontierAnyHitRows(scene, args, "vf", vf, vf_capped, vf_envelope_ms,
                                       "FairFrontierLearnedResidualAnyHit+TI", "learned_residual", stpf_weights, vf_truth),
             load_prepare_ms, 0.0);
    WriteRow(out, args, scene,
             RunFairFrontierAnyHitRows(scene, args, "vf", vf, vf_capped, vf_envelope_ms,
                                       "FairFrontierRandomAnyHit+TI", "random", stpf_weights, vf_truth),
             load_prepare_ms, 0.0);
    WriteRow(out, args, scene,
             RunFairFrontierAnyHitRows(scene, args, "vf", vf, vf_capped, vf_envelope_ms,
                                       "FairFrontierProximityAnyHit+TI", "proximity", stpf_weights, vf_truth),
             load_prepare_ms, 0.0);
    WriteRow(out, args, scene,
             RunFairFrontierAnyHitRows(scene, args, "vf", vf, vf_capped, vf_envelope_ms,
                                       "FairFrontierMotionAnyHit+TI", "motion", stpf_weights, vf_truth),
             load_prepare_ms, 0.0);
    WriteRow(out, args, scene,
             RunOptimizedLearnedAnyHitRows(scene, args, "vf", vf, vf_capped, vf_envelope_ms,
                                           stpf_weights, vf_truth),
             load_prepare_ms, 0.0);
    WriteRow(out, args, scene,
             RunScheduledAnyHitRows(scene, args, "vf", vf, vf_capped, vf_envelope_ms,
                                     "FrozenLearnedAnyHit+TI", "learned", stpf_weights, vf_truth),
             load_prepare_ms, 0.0);
    WriteRow(out, args, scene,
             RunScheduledAnyHitRows(scene, args, "vf", vf, vf_capped, vf_envelope_ms,
                                    "LearnedResidualAnyHit+TI", "learned_residual", stpf_weights, vf_truth),
             load_prepare_ms, 0.0);
    WriteRow(out, args, scene,
             RunScheduledAnyHitRows(scene, args, "vf", vf, vf_capped, vf_envelope_ms,
                                    "RandomAnyHit+TI", "random", stpf_weights, vf_truth),
             load_prepare_ms, 0.0);
    WriteRow(out, args, scene,
             RunScheduledAnyHitRows(scene, args, "vf", vf, vf_capped, vf_envelope_ms,
                                    "ProximityHeuristicAnyHit+TI", "proximity", stpf_weights, vf_truth),
             load_prepare_ms, 0.0);
    WriteRow(out, args, scene,
             RunScheduledAnyHitRows(scene, args, "vf", vf, vf_capped, vf_envelope_ms,
                                    "MotionHeuristicAnyHit+TI", "motion", stpf_weights, vf_truth),
             load_prepare_ms, 0.0);

    const auto ee_begin = std::chrono::steady_clock::now();
    bool ee_capped = false;
    std::vector<Candidate> ee = BuildEECandidates(scene, args.max_ee_candidates, &ee_capped);
    const auto ee_end = std::chrono::steady_clock::now();
    const double ee_envelope_ms = std::chrono::duration<double, std::milli>(ee_end - ee_begin).count();
    std::unordered_map<std::uint64_t, GroupTruth> ee_truth;
    WriteRow(out, args, scene, RunAllExactRows(scene, args, "ee", ee, ee_capped, ee_envelope_ms, &ee_truth, feature_stream), load_prepare_ms, 0.0);
    WriteRow(out, args, scene,
             RunFairFrontierAnyHitRows(scene, args, "ee", ee, ee_capped, ee_envelope_ms,
                                        "FairFrontierLearnedAnyHit+TI", "learned", stpf_weights, ee_truth),
             load_prepare_ms, 0.0);
    WriteRow(out, args, scene,
             RunFairFrontierAnyHitRows(scene, args, "ee", ee, ee_capped, ee_envelope_ms,
                                       "FairFrontierLearnedResidualAnyHit+TI", "learned_residual", stpf_weights, ee_truth),
             load_prepare_ms, 0.0);
    WriteRow(out, args, scene,
             RunFairFrontierAnyHitRows(scene, args, "ee", ee, ee_capped, ee_envelope_ms,
                                       "FairFrontierRandomAnyHit+TI", "random", stpf_weights, ee_truth),
             load_prepare_ms, 0.0);
    WriteRow(out, args, scene,
             RunFairFrontierAnyHitRows(scene, args, "ee", ee, ee_capped, ee_envelope_ms,
                                       "FairFrontierProximityAnyHit+TI", "proximity", stpf_weights, ee_truth),
             load_prepare_ms, 0.0);
    WriteRow(out, args, scene,
             RunFairFrontierAnyHitRows(scene, args, "ee", ee, ee_capped, ee_envelope_ms,
                                       "FairFrontierMotionAnyHit+TI", "motion", stpf_weights, ee_truth),
             load_prepare_ms, 0.0);
    WriteRow(out, args, scene,
             RunOptimizedLearnedAnyHitRows(scene, args, "ee", ee, ee_capped, ee_envelope_ms,
                                           stpf_weights, ee_truth),
             load_prepare_ms, 0.0);
    WriteRow(out, args, scene,
             RunScheduledAnyHitRows(scene, args, "ee", ee, ee_capped, ee_envelope_ms,
                                     "FrozenLearnedAnyHit+TI", "learned", stpf_weights, ee_truth),
             load_prepare_ms, 0.0);
    WriteRow(out, args, scene,
             RunScheduledAnyHitRows(scene, args, "ee", ee, ee_capped, ee_envelope_ms,
                                    "LearnedResidualAnyHit+TI", "learned_residual", stpf_weights, ee_truth),
             load_prepare_ms, 0.0);
    WriteRow(out, args, scene,
             RunScheduledAnyHitRows(scene, args, "ee", ee, ee_capped, ee_envelope_ms,
                                    "RandomAnyHit+TI", "random", stpf_weights, ee_truth),
             load_prepare_ms, 0.0);
    WriteRow(out, args, scene,
             RunScheduledAnyHitRows(scene, args, "ee", ee, ee_capped, ee_envelope_ms,
                                    "ProximityHeuristicAnyHit+TI", "proximity", stpf_weights, ee_truth),
             load_prepare_ms, 0.0);
    WriteRow(out, args, scene,
             RunScheduledAnyHitRows(scene, args, "ee", ee, ee_capped, ee_envelope_ms,
                                    "MotionHeuristicAnyHit+TI", "motion", stpf_weights, ee_truth),
             load_prepare_ms, 0.0);
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << "\n";
    return 1;
  }
}
