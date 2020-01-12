#pragma warning(push, 0)
#include<unordered_set>

#include<Eigen/Core>

#include<g2o/core/block_solver.h>
#include<g2o/core/robust_kernel_impl.h>
#include<g2o/core/optimization_algorithm_levenberg.h>
#include<g2o/solvers/eigen/linear_solver_eigen.h>
#include<g2o/solvers/dense/linear_solver_dense.h>
#include<g2o/types/sba/types_six_dof_expmap.h>
#include<g2o/types/sim3/types_seven_dof_expmap.h>
#pragma warning(pop)

#include"converter.hpp"
#include"tracking/optimizer.hpp"

namespace slam {
namespace optimizer {

void globalBundleAdjustment(std::shared_ptr<Map> map, int iterations) {
    auto keyframes = map->getKeyframes();
    auto mappoints = map->getMappoints();
    std::cout
        << "[optimization] Map contains "
        << keyframes.size() << " keyframes and "
        << mappoints.size() << " mappoints" << std::endl;
    // Set optimizer.
    auto algorithm = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<g2o::BlockSolverX>(
            g2o::make_unique<g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>>()
        )
    );
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(algorithm);
    // Set KeyFrame vertices.
    int kId, mId = 0, mIdT;
    for (const auto& keyframe : keyframes) {
        auto vertex = new g2o::VertexSE3Expmap();
        kId = static_cast<int>(keyframe->id);
        vertex->setId(kId);
        vertex->setFixed(kId == 0);
        vertex->setEstimate(matToSE3Quat(keyframe->getPose()));

        optimizer.addVertex(vertex);
        if (mId < kId) mId = kId;
    }
    mId++;
    mIdT = mId;
    // Set MapPoint vertices.
    for (const auto& mappoint : mappoints) {
        auto point = new g2o::VertexSBAPointXYZ();
        point->setId(mId);
        point->setMarginalized(true);
        point->setEstimate(pointToVec3d(mappoint->getWorldPos()));

        optimizer.addVertex(point);
        // Set edges.
        // Each edge connects current mappoint vertex with
        // every keyframe vertex, that it is visible from.
        // With edge's observation being a keyframe's keypoint.
        auto observations = mappoint->getObservations();
        for (const auto& [keyframe, keypointId] : observations) {
            auto keypoint = keyframe->getFrame()->undistortedKeypoints[keypointId];
            auto edge = new g2o::EdgeSE3ProjectXYZ();
            auto kernel = new g2o::RobustKernelHuber();

            Eigen::Matrix<double, 2, 1> observation;
            observation << keypoint.pt.x, keypoint.pt.y;

            edge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
                optimizer.vertex(mId)
            ));
            edge->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
                optimizer.vertex(static_cast<int>(keyframe->id))
            ));
            edge->setMeasurement(observation);
            edge->setInformation(
                Eigen::Matrix2d::Identity()
                * keyframe->getFrame()->invSigma[keypoint.octave]
            );
            edge->setRobustKernel(kernel);

            edge->fx = keyframe->getFrame()->cameraMatrix->at<float>(0, 0);
            edge->fy = keyframe->getFrame()->cameraMatrix->at<float>(1, 1);
            edge->cx = keyframe->getFrame()->cameraMatrix->at<float>(0, 2);
            edge->cy = keyframe->getFrame()->cameraMatrix->at<float>(1, 2);

            optimizer.addEdge(edge);
        }
        mId++;
    }
    optimizer.setVerbose(false);
    optimizer.initializeOptimization();
    optimizer.optimize(iterations);
    // Update map with optimized hypergraph.
    for (auto& keyframe : keyframes) {
        auto vertex = dynamic_cast<g2o::VertexSE3Expmap*>(
            optimizer.vertex(static_cast<int>(keyframe->id))
        );
        keyframe->setPose(se3QuatToMat(vertex->estimate()));
    }
    for (auto& mappoint : mappoints) {
        auto vertex = dynamic_cast<g2o::VertexSBAPointXYZ*>(
            optimizer.vertex(mIdT++)
        );
        mappoint->setWorldPos(vec3dToPoint3f(vertex->estimate()));
    }
}

void poseOptimization(std::shared_ptr<KeyFrame> keyframe, int iterations) {
    auto algorithm = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<g2o::BlockSolverX>(
            g2o::make_unique<g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>>()
        )
    );
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(algorithm);

    // Set keyframe vertex.
    auto vertex = new g2o::VertexSE3Expmap();
    vertex->setEstimate(matToSE3Quat(keyframe->getPose()));
    vertex->setFixed(false);
    vertex->setId(0);
    optimizer.addVertex(vertex);

    auto edgeVertex = dynamic_cast<g2o::OptimizableGraph::Vertex*>(
        optimizer.vertex(0)
    );
    // Set mappoints vertices.
    int id = 1;
    for (const auto& [i, p] : keyframe->getMapPoints()) {
        int keypointId = p->getObservations()[keyframe];

        auto kernel = new g2o::RobustKernelHuber();
        auto vertexMP = new g2o::VertexSBAPointXYZ();
        auto keypoint = keyframe->getFrame()->undistortedKeypoints[keypointId];

        vertexMP->setEstimate(pointToVec3d(p->getWorldPos()));
        vertexMP->setFixed(true);
        vertexMP->setId(id);

        Eigen::Matrix<double, 2, 1> observation;
        observation << keypoint.pt.x, keypoint.pt.y;

        optimizer.addVertex(vertexMP);
        // Set edge.
        // Edge connects current mappoint vertex with keyframe vertex.
        auto edge = new g2o::EdgeSE3ProjectXYZ();
        edge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
            optimizer.vertex(id)
        ));
        edge->setVertex(1, edgeVertex);
        edge->setMeasurement(observation);
        edge->setInformation(
            Eigen::Matrix2d::Identity()
            * keyframe->getFrame()->invSigma[keypoint.octave]
        );
        edge->setRobustKernel(kernel);
        edge->setLevel(0);

        edge->fx = keyframe->getFrame()->cameraMatrix->at<float>(0, 0);
        edge->fy = keyframe->getFrame()->cameraMatrix->at<float>(1, 1);
        edge->cx = keyframe->getFrame()->cameraMatrix->at<float>(0, 2);
        edge->cy = keyframe->getFrame()->cameraMatrix->at<float>(1, 2);

        optimizer.addEdge(edge);
        id++;
    }

    optimizer.setVerbose(false);
    optimizer.initializeOptimization();
    optimizer.optimize(iterations);
    // Recover optimized pose.
    auto keyframeVertex = dynamic_cast<g2o::VertexSE3Expmap*>(
        optimizer.vertex(0)
    );
    keyframe->setPose(se3QuatToMat(keyframeVertex->estimate()));
}

void localOptimization(std::shared_ptr<KeyFrame> keyframe, int iterations) {
    // Set optimizer.
    auto algorithm = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<g2o::BlockSolverX>(
            g2o::make_unique<g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>>()
        )
    );
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(algorithm);

    std::unordered_set<std::shared_ptr<KeyFrame>> localKeyframes;
    std::unordered_set<std::shared_ptr<KeyFrame>> fixedKeyframes;

    // Set local KeyFrame.
    localKeyframes.insert(keyframe);
    for (auto [connection, count] : keyframe->connections)
        localKeyframes.insert(connection);
    // Set fixed KeyFrame.
    // Fixed KeyFrames are those, that observe a MapPoint,
    // but are not a local KeyFrames (e.g. not in a connection with `keyframe`).
    for (auto [id, mappoint] : keyframe->mappoints) {
        for (auto [observation, idO] : mappoint->getObservations()) {
            if (localKeyframes.find(observation) != localKeyframes.end())
                continue;
            fixedKeyframes.insert(observation);
        }
    }
    // Create local KeyFrame vertices.
    int kId, mId = 0;
    for (const auto localKeyframe : localKeyframes) {
        kId = static_cast<int>(localKeyframe->id);
        auto vertex = new g2o::VertexSE3Expmap();
        vertex->setId(kId);
        vertex->setFixed(kId == 0);
        vertex->setEstimate(matToSE3Quat(localKeyframe->getPose()));

        optimizer.addVertex(vertex);
        if (mId < kId) mId = kId;
    }
    // Create fixed KeyFrame vertices.
    for (const auto fixedKeyframe : fixedKeyframes) {
        kId = static_cast<int>(fixedKeyframe->id);
        auto vertex = new g2o::VertexSE3Expmap();
        vertex->setId(kId);
        vertex->setFixed(kId == 0);
        vertex->setEstimate(matToSE3Quat(fixedKeyframe->getPose()));

        optimizer.addVertex(vertex);
        if (mId < kId) mId = kId;
    }
    mId++;
    int mIdT = mId;
    // Create MapPoint vertices.
    for (auto [id, mappoint] : keyframe->mappoints) {
        auto point = new g2o::VertexSBAPointXYZ();
        point->setId(mId);
        point->setMarginalized(true);
        point->setEstimate(pointToVec3d(mappoint->getWorldPos()));

        optimizer.addVertex(point);
        // Set edges.
        // Each edge connects current mappoint vertex with
        // every keyframe vertex, that it is visible from.
        // With edge's observation being a keyframe's keypoint.
        auto observations = mappoint->getObservations();
        for (const auto& [keyframeO, keypointId] : observations) {
            auto keypoint = keyframeO->getFrame()->undistortedKeypoints[keypointId];
            auto edge = new g2o::EdgeSE3ProjectXYZ();
            auto kernel = new g2o::RobustKernelHuber();

            Eigen::Matrix<double, 2, 1> observation;
            observation << keypoint.pt.x, keypoint.pt.y;

            edge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
                optimizer.vertex(mId)
            ));
            edge->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
                optimizer.vertex(static_cast<int>(keyframeO->id))
            ));
            edge->setMeasurement(observation);
            edge->setInformation(
                Eigen::Matrix2d::Identity()
                * keyframeO->getFrame()->invSigma[keypoint.octave]
            );
            edge->setRobustKernel(kernel);

            edge->fx = keyframeO->getFrame()->cameraMatrix->at<float>(0, 0);
            edge->fy = keyframeO->getFrame()->cameraMatrix->at<float>(1, 1);
            edge->cx = keyframeO->getFrame()->cameraMatrix->at<float>(0, 2);
            edge->cy = keyframeO->getFrame()->cameraMatrix->at<float>(1, 2);

            optimizer.addEdge(edge);
        }
        mId++;
    }
    optimizer.setVerbose(false);
    optimizer.initializeOptimization();
    optimizer.optimize(iterations);
    // Update map with optimized hypergraph.
    for (auto localKeyframe : localKeyframes) {
        auto vertex = dynamic_cast<g2o::VertexSE3Expmap*>(
            optimizer.vertex(static_cast<int>(localKeyframe->id))
        );
        localKeyframe->setPose(se3QuatToMat(vertex->estimate()));
    }
    for (auto [id, mappoint] : keyframe->mappoints) {
        auto vertex = dynamic_cast<g2o::VertexSBAPointXYZ*>(
            optimizer.vertex(mIdT++)
        );
        mappoint->setWorldPos(vec3dToPoint3f(vertex->estimate()));
    }
}

};
};
