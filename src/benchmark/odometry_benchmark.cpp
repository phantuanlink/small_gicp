#include <small_gicp/util/downsampling.hpp>
#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/benchmark/benchmark.hpp>
#include <small_gicp/benchmark/benchmark_odom.hpp>

#include <ctime>

// Get the current time point
std::string getOutoutFileName() {
  auto now = std::chrono::system_clock::now();
  std::time_t now_time_t = std::chrono::system_clock::to_time_t(now);

  // Convert the time_t object to a tm struct
  std::tm now_tm = *std::localtime(&now_time_t);

  std::stringstream ss;
  ss << std::put_time(&now_tm, "%Y-%m-%d_%H-%M-%S");
  return ss.str();

}

void convertToPositionQuaternion(const std::vector<Eigen::Isometry3d>& traj, std::vector<Eigen::Vector3d>& positions, std::vector<Eigen::Quaterniond>& rotations) {
  for (const auto& pose : traj) {
    // Extract position with double precision
    Eigen::Vector3d position = pose.translation();

    // Extract rotation matrix and convert it to quaternion with double precision
    Eigen::Quaterniond quaternion(pose.linear());

    // Store position and quaternion
    positions.push_back(position);
    rotations.push_back(quaternion);
  }
}

int main(int argc, char** argv) {
  using namespace small_gicp;

  if (argc < 3) {
    std::cout << "USAGE: odometry_benchmark <dataset_path> <output_path> [options]" << std::endl;
    std::cout << "OPTIONS:" << std::endl;
    std::cout << "  --visualize" << std::endl;
    std::cout << "  --begin_index <value> (default: 0)" << std::endl;
    std::cout << "  --number_of_frames <value> (default: -1)" << std::endl;
    std::cout << "  --num_threads <value> (default: 4)" << std::endl;
    std::cout << "  --num_neighbors <value> (default: 20)" << std::endl;
    std::cout << "  --downsampling_resolution <value> (default: 0.25)" << std::endl;
    std::cout << "  --voxel_resolution <value> (default: 2.0)" << std::endl;

    const auto odom_names = odometry_names();
    std::stringstream sst;
    for (size_t i = 0; i < odom_names.size(); i++) {
      if (i) {
        sst << "|";
      }
      sst << odom_names[i];
    }

    std::cout << "  --engine <" << sst.str() << "> (default: small_gicp)" << std::endl;
    return 0;
  }

  const std::string dataset_path = argv[1];
  const std::string output_path = argv[2];

  OdometryEstimationParams params;
  std::string engine = "small_gicp";
  int begin_index = 0;
  int number_of_frames = -1;

  for (int i = 0; i < argc; i++) {
    const std::string arg = argv[i];
    if (arg == "--visualize") {
      params.visualize = true;
    } else if (arg == "--num_threads") {
      params.num_threads = std::stoi(argv[i + 1]);
    } else if (arg == "--num_neighbors") {
      params.num_neighbors = std::stoi(argv[i + 1]);
    } else if (arg == "--downsampling_resolution") {
      params.downsampling_resolution = std::stod(argv[i + 1]);
    } else if (arg == "--voxel_resolution") {
      params.voxel_resolution = std::stod(argv[i + 1]);
    } else if (arg == "--engine") {
      engine = argv[i + 1];
    } else if (arg == "--begin_index") {
      begin_index = std::stoi(argv[i + 1]);
    } else if (arg == "--number_of_frames") {
      number_of_frames = std::stoi(argv[i + 1]);
    } else if (arg.size() >= 2 && arg.substr(0, 2) == "--") {
      std::cerr << "unknown option: " << arg << std::endl;
      return 1;
    }
  }

  std::cout << "SIMD in use=" << Eigen::SimdInstructionSetsInUse() << std::endl;
  std::cout << "dataset_path=" << dataset_path << std::endl;
  std::cout << "output_path=" << output_path << std::endl;
  std::cout << "registration_engine=" << engine << std::endl;
  std::cout << "num_threads=" << params.num_threads << std::endl;
  std::cout << "num_neighbors=" << params.num_neighbors << std::endl;
  std::cout << "downsampling_resolution=" << params.downsampling_resolution << std::endl;
  std::cout << "voxel_resolution=" << params.voxel_resolution << std::endl;
  std::cout << "visualize=" << params.visualize << std::endl;
  std::cout << "begin_index=" << begin_index << std::endl;
  std::cout << "number_of_frames=" << number_of_frames << std::endl;

  std::shared_ptr<OdometryEstimation> odom = create_odometry(engine, params);
  if (odom == nullptr) {
    return 1;
  }

  PointCloudPlyDataset dataset(dataset_path, begin_index, number_of_frames);
  std::cout << "num_frames=" << dataset.points.size() << std::endl;
  std::cout << fmt::format("num_points={} [points]", summarize(dataset.points, [](const auto& pts) { return pts.size(); })) << std::endl;

  auto raw_points = dataset.convert<PointCloud>(true);
  std::vector<Eigen::Isometry3d> traj = odom->estimate(raw_points);

  std::cout << "done!" << std::endl;
  odom->report();


  ///// Writing output
  auto output_file = output_path + "/" + getOutoutFileName() + ".txt";
  std::ofstream ofs(output_file);

// Check if the file was successfully opened
  if (!ofs.is_open()) {
      std::cerr << "Error: Unable to create the file." << std::endl;
      return 1; // Return an error code
  }

  std::vector<Eigen::Vector3d> positions;
  positions.reserve(traj.size());
  std::vector<Eigen::Quaterniond> rotations;
  rotations.reserve(traj.size());

  convertToPositionQuaternion(traj, positions, rotations);

  for (auto i = 0; i < traj.size(); i++) {
    ofs << fmt::format("{:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}", positions[i][0], positions[i][1], positions[i][2], rotations[i].w(), rotations[i].x(), rotations[i].y(), rotations[i].z());
    ofs << std::endl;
  }

  // Close the file
  ofs.close();
  std::cout << "Result saved successfully at "<< output_file << std::endl;
  return 0;
}
