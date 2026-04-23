#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <limits>
#include <queue>
#include <iomanip>

#include "parlaylib/include/parlay/primitives.h"
#include "parlaylib/include/parlay/parallel.h"
#include "parlaylib/include/parlay/sequence.h"
#include "parlaylib/include/parlay/utilities.h"


// A simple 2D point structure
struct Point2D {
  double x, y;
  Point2D(double xx=0.0, double yy=0.0) : x(xx), y(yy) {}
};

// A helper to compute squared distance
inline double squared_distance(const Point2D& a, const Point2D& b) {
  double dx = a.x - b.x;
  double dy = a.y - b.y;
  return dx*dx + dy*dy;
}

// KD-Tree node
struct KDNode {
  int axis;          // 0 for x, 1 for y
  double splitValue; // coordinate pivot
  int pointIndex;    // index in the original array

  KDNode* left;
  KDNode* right;

  KDNode() : axis(0), splitValue(0.0), pointIndex(-1),
             left(nullptr), right(nullptr) {}
};

// DistIndex for storing (distance^2, index)
struct DistIndex {
  double dist;
  int index;
  DistIndex(double d=0, int i=0) : dist(d), index(i) {}
};

// For a max-heap, we want to put the largest distance on top
inline bool operator<(const DistIndex &a, const DistIndex &b) {
  return a.dist < b.dist;
}


KDNode* build_kd_tree(
    parlay::slice<int*, int*> indices,
    const parlay::sequence<Point2D>& points,
    int depth = 0
) {
  size_t n = indices.size();
  if (n == 0) return nullptr;

  int axis = depth % 2;

  parlay::sequence<int> local_indices(indices.begin(), indices.end());
  parlay::sort_inplace(local_indices, [&](int a, int b) {
    double va = (axis == 0) ? points[a].x : points[a].y;
    double vb = (axis == 0) ? points[b].x : points[b].y;
    if (va == vb) return a < b;
    return va < vb;
  });

  size_t mid = n / 2;
  int pivot_idx = local_indices[mid];

  auto* node = new KDNode();
  node->axis = axis;
  node->pointIndex = pivot_idx;
  node->splitValue = (axis == 0) ? points[pivot_idx].x : points[pivot_idx].y;

  auto left_slice = local_indices.cut(0, mid);
  auto right_slice = local_indices.cut(mid + 1, n);

  // Recurse on left and right in parallel
  parlay::par_do(
      [&]() { node->left = build_kd_tree(left_slice, points, depth + 1); },
      [&]() { node->right = build_kd_tree(right_slice, points, depth + 1); });


  return node;
}

class KNNHelper {
public:
  KNNHelper(const parlay::sequence<Point2D>& pts, int kk)
    : points(pts), k(kk) {
    best.reserve(k);
  }

  // Perform recursive search
  void search(const KDNode* node, const Point2D& q) {
    if (node == nullptr) return;

    int idx = node->pointIndex;
    double dist2 = squared_distance(q, points[idx]);
    update_best(dist2, idx);

    double qv = (node->axis == 0) ? q.x : q.y;
    const KDNode* near_side = (qv < node->splitValue) ? node->left : node->right;
    const KDNode* far_side = (qv < node->splitValue) ? node->right : node->left;

    search(near_side, q);

    double boundary_dist2 = (qv - node->splitValue) * (qv - node->splitValue);
    double worst_best_dist2 =
        (best.size() < static_cast<size_t>(k)) ? std::numeric_limits<double>::infinity()
                                               : best.front().dist;
    if (boundary_dist2 <= worst_best_dist2) {
      search(far_side, q);
    }
  }

  // Return final results sorted by ascending distance
  parlay::sequence<DistIndex> get_results() const {
    parlay::sequence<DistIndex> result(best.begin(), best.end());
    parlay::sort_inplace(result, [&](auto &a, auto &b){
      if (a.dist == b.dist) return a.index < b.index;
      return a.dist < b.dist;
    });
    return result;
  }

private:
  const parlay::sequence<Point2D>& points;
  int k;
  std::vector<DistIndex> best; // will be a max-heap

  static bool heap_less(const DistIndex& a, const DistIndex& b) {
    if (a.dist == b.dist) return a.index < b.index;
    return a.dist < b.dist;
  }

  // If we have fewer than k, push. Otherwise compare with largest so far 
  void update_best(double dist2, int idx) {
    DistIndex candidate(dist2, idx);
    if (k <= 0) return;

    if (best.size() < static_cast<size_t>(k)) {
      best.push_back(candidate);
      std::push_heap(best.begin(), best.end(), heap_less);
      return;
    }

    const DistIndex& worst = best.front();
    bool candidate_is_better =
        (candidate.dist < worst.dist) ||
        (candidate.dist == worst.dist && candidate.index < worst.index);

    if (candidate_is_better) {
      std::pop_heap(best.begin(), best.end(), heap_less);
      best.back() = candidate;
      std::push_heap(best.begin(), best.end(), heap_less);
    }
  }
};

// Parallel k-NN for all queries
parlay::sequence<parlay::sequence<DistIndex>>
knn_search_all(const KDNode* root,
               const parlay::sequence<Point2D>& data_points,
               const parlay::sequence<Point2D>& query_points,
               int k) {
  int Q = (int)query_points.size();
  parlay::sequence<parlay::sequence<DistIndex>> results(Q);

  parlay::parallel_for(0, Q, [&](int i){
    KNNHelper helper(data_points, k);
    helper.search(root, query_points[i]);
    results[i] = helper.get_results();
  });

  return results;
}

// A function to load points from a file
parlay::sequence<Point2D> load_points_from_file(const std::string &filename) {
  std::ifstream in(filename);


  int n = 0;
  in >> n;

  parlay::sequence<Point2D> points(n);
  for (int i = 0; i < n; i++) {
    double x = 0.0, y = 0.0;
    in >> x >> y;
    points[i] = Point2D(x, y);
  }
  return points;
}

void free_kd_tree(KDNode* node) {
  if (node == nullptr) return;
  free_kd_tree(node->left);
  free_kd_tree(node->right);
  delete node;
}

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0]
              << " <data_file> <query_file> <k>\n";
    return 1;
  }

  std::string data_file  = argv[1];
  std::string query_file = argv[2];
  int k = std::stoi(argv[3]);

  auto data_points = load_points_from_file(data_file);
  int N = (int)data_points.size();
  if (N == 0) {
    std::cerr << "No data points loaded.\n";
    return 1;
  }
  k = std::max(1, std::min(k, N));

  parlay::sequence<int> indices(N);
  parlay::parallel_for(0, N, [&](int i){ indices[i] = i; });
  KDNode* root = build_kd_tree(indices.cut(0, N), data_points, 0);

  auto query_points = load_points_from_file(query_file);
  int Q = (int)query_points.size();

  auto results = knn_search_all(root, data_points, query_points, k);

  for (int q = 0; q < Q; q++) {
    std::cout << "Query " << q << ": ("
              << std::fixed << std::setprecision(2)
              << query_points[q].x << ", "
              << query_points[q].y << ")\n";
    std::cout << "  kNN: ";
    for (auto &di : results[q]) {
      std::cout << "(dist2=" << std::fixed << std::setprecision(2) << di.dist
                << ", idx=" << di.index << ") ";
    }
    std::cout << "\n";
  }

  free_kd_tree(root);

  return 0;
}
