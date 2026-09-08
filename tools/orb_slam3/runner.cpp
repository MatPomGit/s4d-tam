// Frame-wise ORB-SLAM3 monocular runner for the S4D-TAM benchmark.
// This source links against ORB-SLAM3 (GPL-3.0); any distributed combined binary
// must comply with the ORB-SLAM3 GPL-3.0 license terms.

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <System.h>

struct FrameSpec {
    double timestamp{};
    std::string relative_path;
};

static std::vector<FrameSpec> load_rgb(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open rgb.txt: " + path);
    }
    std::vector<FrameSpec> frames;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::istringstream stream(line);
        FrameSpec frame;
        if (!(stream >> frame.timestamp >> frame.relative_path)) {
            throw std::runtime_error("invalid rgb.txt row: " + line);
        }
        frames.push_back(frame);
    }
    if (frames.empty()) {
        throw std::runtime_error("rgb.txt contains no frames");
    }
    for (std::size_t i = 1; i < frames.size(); ++i) {
        if (!(frames[i].timestamp > frames[i - 1].timestamp)) {
            throw std::runtime_error("rgb.txt timestamps are not strictly increasing");
        }
    }
    return frames;
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "Usage: s4dtam_orb_slam3_runner <vocabulary> <settings> <sequence_dir> <output_csv>\n";
        return 2;
    }

    const std::string vocabulary = argv[1];
    const std::string settings = argv[2];
    const std::string sequence_dir = argv[3];
    const std::string output_csv = argv[4];

    try {
        const auto frames = load_rgb(sequence_dir + "/rgb.txt");
        std::ofstream output(output_csv);
        if (!output) {
            throw std::runtime_error("cannot create output CSV: " + output_csv);
        }
        output << "timestamp,tx,ty,tz,qx,qy,qz,qw,latency_ms,tracking_valid\n";
        output << std::setprecision(17);

        ORB_SLAM3::System slam(vocabulary, settings, ORB_SLAM3::System::MONOCULAR, false);

        Sophus::SE3f last_valid;
        bool have_valid_pose = false;
        for (const auto& frame : frames) {
            const std::string image_path = sequence_dir + "/" + frame.relative_path;
            cv::Mat image = cv::imread(image_path, cv::IMREAD_UNCHANGED);
            if (image.empty()) {
                slam.Shutdown();
                throw std::runtime_error("cannot load image: " + image_path);
            }

            const auto begin = std::chrono::steady_clock::now();
            Sophus::SE3f tcw = slam.TrackMonocular(image, frame.timestamp);
            const auto end = std::chrono::steady_clock::now();
            const double latency_ms =
                std::chrono::duration<double, std::milli>(end - begin).count();

            const int state = slam.GetTrackingState();
            const bool valid = state == 2;  // Tracking::OK
            Sophus::SE3f twc;
            if (valid) {
                twc = tcw.inverse();
                last_valid = twc;
                have_valid_pose = true;
            } else if (have_valid_pose) {
                twc = last_valid;
            } else {
                twc = Sophus::SE3f();
            }

            const Eigen::Vector3f t = twc.translation();
            const Eigen::Quaternionf q = twc.unit_quaternion();
            output << frame.timestamp << ',' << t.x() << ',' << t.y() << ',' << t.z() << ','
                   << q.x() << ',' << q.y() << ',' << q.z() << ',' << q.w() << ','
                   << latency_ms << ',' << (valid ? 1 : 0) << '\n';
        }

        slam.Shutdown();
        output.flush();
        if (!output) {
            throw std::runtime_error("failed while writing output CSV");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "s4dtam_orb_slam3_runner: " << error.what() << '\n';
        return 1;
    }
}
