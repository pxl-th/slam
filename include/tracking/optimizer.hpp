#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include"map/map.hpp"

namespace slam {
/**
 * Namespace contains functions for performing optimizations.
 */
namespace optimizer {

/**
 * Perform Global Bundle Adjustment on the map.
 * **Note** this updates map in-place.
 *
 * Builds a hypergraph out of keyframes and mappoints. \n
 * With estimates of the vertices being keyframe's poses
 * and mappoint's world positions. \n
 * Each hyperdge in hypergraph connects mappoing vertex with each
 * keyframe vertex that it is visible from, with edge's information
 * being a keyframe's keypoint. \n
 * After performing optimization, map is updated with optimized
 * vertices' values.
 *
 * @param map Map on which to perform BA.
 * @param iterations Number of iterations to perform optimization.
 */
void globalBundleAdjustment(std::shared_ptr<Map> map, int iterations);

};
};

#endif
