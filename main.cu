#include "dbscan-fast.cuh"

#include <vector>
#include <map>
#include <random>
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;
typedef Point<float, 2> Point2D ;

// Generate test 2D data with 3 clusters
std::vector<Point2D> generateTestData(int pointsPerCluster = 1500) {
    std::default_random_engine gen;
    std::normal_distribution<double> dist(0.0, 0.5);

    std::vector<std::pair<double, double>> centers = {
        {0, 0}, {10, 10}, {-10, -10}
    };

    std::vector<Point2D> points;
    for (const auto& [cx, cy] : centers) {
        for (int i = 0; i < pointsPerCluster; ++i) {
            Point2D p;
            p.data.push_back(cx + dist(gen));
            p.data.push_back(cy + dist(gen) );
            points.push_back(p);
        }
    }

    return points;
}

// Plot 2D clusters using matplotlib-cpp
void plot2DClusters(const std::vector<Point2D>& points, DBSCAN &scanner) {
    std::map<int, std::vector<double>> xs, ys;

    for(int i=0;i < points.size(); ++i){
        xs[scanner.label(i)].push_back(points[i].data[0]);
        ys[scanner.label(i)].push_back(points[i].data[1]);
    }

    plt::figure();
    plt::title("2D DBSCAN Clustering");

    for (const auto& [clusterID, x_vals] : xs) {
        const auto& y_vals = ys[clusterID];
        if(y_vals.size() < 10)continue;
        std::string label = (clusterID == -1) ? "Noise" : "Cluster " + std::to_string(clusterID);
        plt::scatter(x_vals, y_vals, 10.0, {{"label", label}});
    }

    plt::legend();
    plt::show();
}

float f(float val){
    return val;
}

int main() {
    auto data = generateTestData();
    DBSCAN scanner(0.3, 10);
    scanner.identify_cluster(data, f);
    scanner.show_labels();
    plot2DClusters(data, scanner);
    return 0;
}
