#include <Core/UniqueIDGenerator.h>

#include <Math/Transform.h>
#include <Math/Color.h>
#include <Math/Conversions.h>
#include <Math/Matrix.h>
#include <Math/Quaternion.h>
#include <Math/Vector.h>

#include <Scene/SceneNode.h>

#include <cstdio>
#include <iostream>

using namespace Cogwheel::Core;
using namespace Cogwheel::Math;
using namespace Cogwheel::Scene; 
using std::cout;
using std::endl;

void main(int argc, char** argv) {

    Vector3f v0 = Vector3f(3, 1, 7);
    cout << "v0: " << v0 << endl;

    Quaternionf q0 = Quaternionf::fromAngleAxis(20.0f, normalize(Vector3f(6,4,7)));
    cout << "q0: " << q0 << endl;
    cout << "q0 * v0: " << (q0 * v0) << endl;

    Quaternionf q1 = Quaternionf(-q0.x, -q0.y, -q0.z, -q0.w);
    cout << "q1: " << q1 << endl;
    cout << "q1 * v0: " << (q1 * v0) << endl;

}
