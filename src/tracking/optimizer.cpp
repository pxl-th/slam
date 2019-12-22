#pragma warning(push, 0)
#include<Eigen/Core>

#include<g2o/core/block_solver.h>
#include<g2o/core/optimization_algorithm_levenberg.h>
#include<g2o/solvers/eigen/linear_solver_eigen.h>
#include<g2o/solvers/dense/linear_solver_dense.h>
#include<g2o/types/sba/types_six_dof_expmap.h>
#include<g2o/types/sim3/types_seven_dof_expmap.h>
#pragma warning(pop)

#include"converter.hpp"
#include"tracking/optimizer.hpp"

namespace slam {

void Optimizer::globalBundleAdjustment(
    std::shared_ptr<Map> map, int iterations
) {
    // Set optimizer.
    auto algorithm = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<g2o::BlockSolverX>(
            g2o::make_unique<g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>>()
        )
    );
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(algorithm);

    auto keyframes = map->getKeyframes();
    auto mappoints = map->getMappoints();

    // Temporarily store pointers using smart pointers.
    int kId, mId = 0;
    std::vector<std::shared_ptr<g2o::VertexSE3Expmap>> keyframeVertices;
    std::vector<std::shared_ptr<g2o::VertexSBAPointXYZ>> mappointVertices;
    std::vector<std::shared_ptr<g2o::EdgeSE3ProjectXYZ>> hyperEdges;

    // Set KeyFrame vertices.
    for (const auto& keyframe : keyframes) {
        auto vertex = std::shared_ptr<g2o::VertexSE3Expmap>(
            new g2o::VertexSE3Expmap()
        );
        kId = static_cast<int>(keyframe->id);
        vertex->setId(kId);
        vertex->setFixed(kId == 0);
        vertex->setEstimate(matToSE3Quat(keyframe->getPose()));

        keyframeVertices.push_back(vertex);
        optimizer.addVertex(vertex.get());
        if (mId < kId) mId = kId;
    }
    mId++;

    // Set MapPoint vertices.
    for (const auto& mappoint : mappoints) {
        auto point = std::shared_ptr<g2o::VertexSBAPointXYZ>(
            new g2o::VertexSBAPointXYZ()
        );
        point->setId(mId++);
        point->setMarginalized(true);
        point->setEstimate(pointToVec3d(mappoint->getWorldPos()));

        mappointVertices.push_back(point);
        optimizer.addVertex(point.get());

        // Set edges.
        auto observations = mappoint->getObservations();
        for (const auto& [keyframe, keypointId] : observations) {
            auto keypoint = keyframe->getFrame().keypoints[keypointId];
            Eigen::Matrix<double, 2, 1> observation;
            observation << keypoint.pt.x, keypoint.pt.y;

            auto edge = std::shared_ptr<g2o::EdgeSE3ProjectXYZ>(
                new g2o::EdgeSE3ProjectXYZ()
            );

            edge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
                optimizer.vertex(mId)
            ));
            edge->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
                optimizer.vertex(keyframe->id)
            ));
            edge->setMeasurement(observation);
            /**
             * TODO:
             * - calculate inverted sigma for pyramid levels (octaves)
             * - set hubert constant
             */
        }
    }

    /**
     * MapPoint's observations should store:
     * {keyframe: id of the keypoint that corresponds to current mappoint}
     *
     * Edge conntects:
     * Current mappoint id with every keyframe id that is in mappoint's observations
     */
    delete algorithm;
}

};
