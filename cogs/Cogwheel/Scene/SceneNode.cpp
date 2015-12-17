// Cogwheel scene node.
// ---------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include "SceneNode.h"

namespace Cogwheel {
namespace Scene {

SceneNodes::UIDGenerator SceneNodes::mUIDGenerator = UIDGenerator(0u);
std::string* SceneNodes::mNames = nullptr;

SceneNodes::UID* SceneNodes::mParentIDs = nullptr;
SceneNodes::UID* SceneNodes::mSiblingIDs = nullptr;
SceneNodes::UID* SceneNodes::mFirstChildIDs = nullptr;

Math::Transform* SceneNodes::mGlobalTransforms = nullptr;

void SceneNodes::allocate(unsigned int capacity) {
    if (isAllocated())
        return;

    mUIDGenerator = UIDGenerator(capacity);
    capacity = mUIDGenerator.capacity();
    
    mNames = new std::string[capacity];
    
    mParentIDs = new SceneNodes::UID[capacity];
    mSiblingIDs = new SceneNodes::UID[capacity];
    mFirstChildIDs = new SceneNodes::UID[capacity];

    mGlobalTransforms = new Math::Transform[capacity];

    // Allocate dummy element at 0.
    mNames[0] = "Dummy Node";
    mParentIDs[0] = mFirstChildIDs[0] = mSiblingIDs[0] = UID::InvalidUID();
    mGlobalTransforms[0] = Math::Transform::identity();
}

void SceneNodes::deallocate() {
    if (!isAllocated())
        return;

    mUIDGenerator = UIDGenerator(0u);
    delete[] mNames; mNames = nullptr;

    delete[] mParentIDs; mParentIDs = nullptr;
    delete[] mSiblingIDs; mSiblingIDs = nullptr;
    delete[] mFirstChildIDs; mFirstChildIDs = nullptr;

    delete[] mGlobalTransforms; mGlobalTransforms = nullptr;
}

void SceneNodes::reserve(unsigned int newCapacity) {
    unsigned int oldCapacity = capacity();
    mUIDGenerator.reserve(newCapacity);
    reserveNodeData(newCapacity, oldCapacity);
}

template <typename T>
static inline T* resizeAndCopyArray(T* oldArray, unsigned int newCapacity, unsigned int copyableElements) {
    T* newArray = new T[newCapacity];
    std::copy(oldArray, oldArray + copyableElements, newArray);
    delete[] oldArray;
    return newArray;
}

void SceneNodes::reserveNodeData(unsigned int newCapacity, unsigned int oldCapacity) {
    const unsigned int copyableElements = newCapacity < oldCapacity ? newCapacity : oldCapacity;

    mNames = resizeAndCopyArray(mNames, newCapacity, copyableElements);

    mParentIDs = resizeAndCopyArray(mParentIDs, newCapacity, copyableElements);
    mSiblingIDs = resizeAndCopyArray(mSiblingIDs, newCapacity, copyableElements);
    mFirstChildIDs = resizeAndCopyArray(mFirstChildIDs, newCapacity, copyableElements);

    mGlobalTransforms = resizeAndCopyArray(mGlobalTransforms, newCapacity, copyableElements);
}

SceneNodes::UID SceneNodes::create(const std::string& name) {
    unsigned int oldCapacity = mUIDGenerator.capacity();
    UID id = mUIDGenerator.generate();
    if (oldCapacity != mUIDGenerator.capacity())
        // The capacity has changed and the size of all arrays need to be adjusted.
        reserveNodeData(mUIDGenerator.capacity(), oldCapacity);

    mNames[id] = name;
    mParentIDs[id] = mFirstChildIDs[id] = mSiblingIDs[id] = UID::InvalidUID();
    mGlobalTransforms[id] = Math::Transform::identity();
    return id;
}

void SceneNodes::setParent(SceneNodes::UID nodeID, const SceneNodes::UID parentID) {
    SceneNodes::UID oldParentID = mParentIDs[nodeID];
    if (nodeID != parentID && nodeID != UID::InvalidUID()) {
        // Detach current node from it's place in the hierarchy
        if (oldParentID != UID::InvalidUID()) {
            // Find old previous sibling.
            UID sibling = mFirstChildIDs[oldParentID];
            if (sibling == nodeID) {
                mFirstChildIDs[oldParentID] = mSiblingIDs[sibling];
            } else {
                while (nodeID != mSiblingIDs[sibling])
                    sibling = mSiblingIDs[sibling];
                mSiblingIDs[sibling] = mSiblingIDs[nodeID];
            }
        }

        // Attach it to the new parent as the first child and link it to the other siblings.
        mParentIDs[nodeID] = parentID;
        mSiblingIDs[nodeID] = mFirstChildIDs[parentID];
        mFirstChildIDs[parentID] = nodeID;
    }
}

std::vector<SceneNodes::UID> SceneNodes::getSiblingIDs(SceneNodes::UID nodeID) {
    SceneNodes::UID parentID = mParentIDs[nodeID];
    
    return getChildrenIDs(parentID);
}

std::vector<SceneNodes::UID> SceneNodes::getChildrenIDs(SceneNodes::UID nodeID) {
    std::vector<SceneNodes::UID> res(0);
    SceneNodes::UID child = mFirstChildIDs[nodeID];
    while (child != UID::InvalidUID()) {
        res.push_back(child);
        child = mSiblingIDs[child];
    }
    return res;
}

bool SceneNodes::hasChild(SceneNodes::UID nodeID, SceneNodes::UID testedChildID) {
    SceneNodes::UID child = mFirstChildIDs[nodeID];
    while (child != UID::InvalidUID()) {
        if (child == testedChildID)
            return true;
        child = mSiblingIDs[child];
    }
    return false;
}

std::vector<SceneNode> SceneNode::getChildren() const {
    std::vector<SceneNodes::UID> childrenIDs = SceneNodes::getChildrenIDs(mID);
    std::vector<SceneNode> children;
    children.reserve(childrenIDs.size());
    for (SceneNodes::UID id : childrenIDs)
        children.push_back(SceneNode(id));
    return children;
}

Math::Transform SceneNodes::getLocalTransform(SceneNodes::UID nodeID) {
    UID parentID = mParentIDs[nodeID];
    const Math::Transform parentTransform = mGlobalTransforms[parentID];
    const Math::Transform transform = mGlobalTransforms[nodeID];
    return inverse(parentTransform) * transform; // TODO Move delta transform computation into method.
}

void SceneNodes::setLocalTransform(SceneNodes::UID nodeID, Math::Transform transform) {
    if (nodeID == UID::InvalidUID()) return;

    // Update global transform.
    UID parentID = mParentIDs[nodeID];
    Math::Transform parentTransform = mGlobalTransforms[parentID];
    Math::Transform newGlobalTransform = parentTransform * transform;
    Math::Transform deltaTransform = inverse(mGlobalTransforms[nodeID]) * newGlobalTransform; // TODO Move into delta method.
    mGlobalTransforms[nodeID] = newGlobalTransform;

    // Update global transforms of all children.
    traverseAllChildren(nodeID, [=](SceneNodes::UID childID) {
        mGlobalTransforms[childID] = deltaTransform * mGlobalTransforms[childID];
    });
}

void SceneNodes::setGlobalTransform(SceneNodes::UID nodeID, Math::Transform transform) {
    if (nodeID == UID::InvalidUID()) return;
    
    Math::Transform deltaTransform = inverse(mGlobalTransforms[nodeID]) * transform; // TODO Move into delta method.
    mGlobalTransforms[nodeID] = transform;

    // Update global transforms of all children.
    traverseAllChildren(nodeID, [=](SceneNodes::UID childID) {
        mGlobalTransforms[childID] = deltaTransform * mGlobalTransforms[childID];
    });
}

} // NS Scene
} // NS Cogwheel