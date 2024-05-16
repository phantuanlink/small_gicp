// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
#pragma once

#include <chrono>
#include <deque>
#include <vector>
#include <numeric>
#include <algorithm>
#include <filesystem>
#include <Eigen/Core>
#include <fmt/format.h>
#include <small_gicp/points/traits.hpp>
#include <small_gicp/benchmark/read_points.hpp>

namespace small_gicp {

struct Stopwatch {
public:
  void start() { t1 = t2 = std::chrono::high_resolution_clock::now(); }
  void stop() { t2 = std::chrono::high_resolution_clock::now(); }
  void lap() {
    t1 = t2;
    t2 = std::chrono::high_resolution_clock::now();
  }

  double sec() const { return std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e9; }
  double msec() const { return std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e6; }

public:
  std::chrono::high_resolution_clock::time_point t1;
  std::chrono::high_resolution_clock::time_point t2;
};

struct Summarizer {
public:
  Summarizer(bool full_log = false) : full_log(full_log), num_data(0), sum(0.0), sq_sum(0.0), last_x(0.0) {}

  void push(double x) {
    num_data++;
    sum += x;
    sq_sum += x * x;
    last_x = x;

    full_data.emplace_back(x);
  }

  std::pair<double, double> mean_std() const {
    const double mean = sum / num_data;
    const double var = (sq_sum - mean * sum) / num_data;
    return {mean, std::sqrt(var)};
  }

  double median() const {
    if (!full_log || full_data.empty()) {
      return std::nan("");
    }

    std::vector<double> sorted(full_data.begin(), full_data.end());
    std::nth_element(sorted.begin(), sorted.end(), sorted.begin() + sorted.size() / 2);
    return sorted[sorted.size() / 2];
  }

  double last() const { return last_x; }

  std::string str() const {
    if (full_log) {
      const auto [mean, std] = mean_std();
      const double med = median();
      return fmt::format("{:.3f} +- {:.3f} (median={:.3f})", mean, std, med);
    }

    const auto [mean, std] = mean_std();
    return fmt::format("{:.3f} +- {:.3f} (last={:.3f})", mean, std, last_x);
  }

private:
  bool full_log;
  std::deque<double> full_data;

  size_t num_data;
  double sum;
  double sq_sum;
  double last_x;
};

template <typename Container, typename Transform>
std::string summarize(const Container& container, const Transform& transform) {
  Summarizer summarizer;
  for (auto itr = std::begin(container); itr != std::end(container); itr++) {
    summarizer.push(transform(*itr));
  }
  return summarizer.str();
}

struct KittiDataset {
public:
  explicit KittiDataset(const std::string& dataset_path, size_t max_num_data = 1000000) {
    std::vector<std::string> filenames;
    for (auto path : std::filesystem::directory_iterator(dataset_path)) {
      if (path.path().extension() != ".bin") {
        continue;
      }

      filenames.emplace_back(path.path().string());
    }

    std::sort(filenames.begin(), filenames.end());
    if (filenames.size() > max_num_data) {
      filenames.resize(max_num_data);
    }

    points.resize(filenames.size());
    std::transform(filenames.begin(), filenames.end(), points.begin(), [](const std::string& filename) { return read_points(filename); });
  }

  template <typename PointCloud>
  std::vector<std::shared_ptr<PointCloud>> convert(bool release = false) {
    std::vector<std::shared_ptr<PointCloud>> converted(points.size());
    std::transform(points.begin(), points.end(), converted.begin(), [=](auto& raw_points) {
      auto points = std::make_shared<PointCloud>();
      traits::resize(*points, raw_points.size());
      for (size_t i = 0; i < raw_points.size(); i++) {
        traits::set_point(*points, i, raw_points[i].template cast<double>());
      }

      if (release) {
        raw_points.clear();
        raw_points.shrink_to_fit();
      }
      return points;
    });

    if (release) {
      points.clear();
      points.shrink_to_fit();
    }

    return converted;
  }

public:
  std::vector<std::vector<Eigen::Vector4f>> points;
};

struct PointCloudPlyDataset {
public:

  /*
  * @param dataset_path: path to folder that contains .ply files
  * @param begin_idx: the begin index that we would start of the series of filenames
  * @param number_of_frames: the number of frames to be processed. Default -1, meaning we process all frames starting from the begin_idx
  * @param max_num_data: max amount of of frames
  */
  explicit PointCloudPlyDataset(const std::string& dataset_path, int begin_idx = 0, int number_of_frames = -1, size_t max_num_data = 1000000) {
    std::vector<std::string> filenames;
    for (auto path : std::filesystem::directory_iterator(dataset_path)) {
      if (path.path().extension() != ".ply") {
        continue;
      }
      filenames.emplace_back(path.path().string());
    }

    std::sort(filenames.begin(), filenames.end(), [](const std::string& filename1, const std::string& filename2) {
      size_t pos_extenstion = filename1.find_last_of(".");
      size_t pos_underscore = filename1.find_last_of("_");
      // Extract the filename without extension
      int file_index_1 = std::stoi(filename1.substr(pos_underscore + 1, pos_extenstion));

      pos_extenstion = filename2.find_last_of(".");
      pos_underscore = filename2.find_last_of("_");
      int file_index_2 = std::stoi(filename2.substr(pos_underscore + 1, pos_extenstion));

      return file_index_1 < file_index_2;
    });

    if (filenames.size() > max_num_data) {
      filenames.resize(max_num_data);
    }

    std::cout << "Found " << filenames.size() << " ply files\n";
    for (auto& filename : filenames) {
      std::cout << filename << std::endl;
    }

    if (number_of_frames == -1) {
      number_of_frames = filenames.size();
    } else if (filenames.size() < (begin_idx + number_of_frames)) {
      number_of_frames = (begin_idx + number_of_frames) - filenames.size();
    }
    std::cout << "About to read the first " << number_of_frames << " frames\n";  // Extract the first 10 elements

    std::vector<std::string> filenames_for_registration(filenames.begin() + begin_idx, filenames.begin() + begin_idx + number_of_frames);

    points.resize(filenames_for_registration.size());
    std::transform(filenames_for_registration.begin(), filenames_for_registration.end(), points.begin(), [](const std::string& filename) { return read_any_ply(filename); });
  }

  template <typename PointCloud>
  std::vector<std::shared_ptr<PointCloud>> convert(bool release = false) {
    std::vector<std::shared_ptr<PointCloud>> converted(points.size());
    std::transform(points.begin(), points.end(), converted.begin(), [=](auto& raw_points) {
      auto points = std::make_shared<PointCloud>();
      traits::resize(*points, raw_points.size());
      for (size_t i = 0; i < raw_points.size(); i++) {
        traits::set_point(*points, i, raw_points[i].template cast<double>());
      }

      if (release) {
        raw_points.clear();
        raw_points.shrink_to_fit();
      }
      return points;
    });

    if (release) {
      points.clear();
      points.shrink_to_fit();
    }

    return converted;
  }

public:
  std::vector<std::vector<Eigen::Vector4f>> points;
};

}  // namespace small_gicp
