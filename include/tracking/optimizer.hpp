#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include"map/map.hpp"

namespace slam {

class Optimizer {
public:
    static void globalBundleAdjustment(std::shared_ptr<Map> map, int iterations);
};

};

#endif
