// Cogwheel scene node.
// ---------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _COGWHEEL_SCENE_SCENE_NODE_H_
#define _COGWHEEL_SCENE_SCENE_NODE_H_

#include <Core/UniqueIDGenerator.h>
#include <Math/Transform.h>

#include <vector>


namespace Cogwheel {
namespace Scene {

// ---------------------------------------------------------------------------
// Container class for the cogwheel scene node.
// TODO 
// * Allocate all (or most) internal arrays in one big chunk.
// * In place iteration over siblings and children and apply a lambda/closure/functor.
// * Read/write lock on all scene nodes(multiple readers obviously)
// * Change the sibling/children layout, so sibling IDs or perhaps siblings are always allocated next too each other?
//  * Requires an extra indirection though, since node ID's wonøt match the node positions anymore.
//  * Then I could use Core::Array to quickly construct a list of children.
//  * Could be done (partially) at the end of the main loop.
// ---------------------------------------------------------------------------
class SceneNodes final {
public:
    typedef Core::TypedUIDGenerator<SceneNodes> UIDGenerator;
    typedef UIDGenerator::UID UID;

    static bool isAllocated() { return mGlobalTransforms != nullptr; }
    static void allocate(unsigned int capacity);
    static void deallocate();

    static inline unsigned int capacity() { return mUIDGenerator.capacity(); }
    static void reserve(unsigned int capacity);
    static bool has(SceneNodes::UID nodeID) { return mUIDGenerator.has(nodeID); }

    static SceneNodes::UID create(const std::string& name);

    static inline std::string getName(SceneNodes::UID nodeID) { return mNames[nodeID]; }
    static inline void setName(SceneNodes::UID nodeID, const std::string& name) { mNames[nodeID] = name; }

    static inline SceneNodes::UID getParentID(SceneNodes::UID nodeID) { return mParentIDs[nodeID]; }
    static void setParent(SceneNodes::UID nodeID, const SceneNodes::UID parentID);
    static bool hasChild(SceneNodes::UID nodeID, SceneNodes::UID testedChildID);
    static std::vector<SceneNodes::UID> getSiblingIDs(SceneNodes::UID nodeID);
    static std::vector<SceneNodes::UID> getChildrenIDs(SceneNodes::UID nodeID);

    static Math::Transform getLocalTransform(SceneNodes::UID nodeID);
    static void setLocalTransform(SceneNodes::UID nodeID, Math::Transform transform);
    static Math::Transform getGlobalTransform(SceneNodes::UID nodeID) { return mGlobalTransforms[nodeID];}
    static void setGlobalTransform(SceneNodes::UID nodeID, Math::Transform transform);

    template<typename F>
    static void traverseGraph(SceneNodes::UID nodeID, F& function);
    template<typename F>
    static void traverseAllChildren(SceneNodes::UID nodeID, F& function);

private:
    static void reserveNodeData(unsigned int capacity, unsigned int oldCapacity);

    static UIDGenerator mUIDGenerator;
    static std::string* mNames;

    static SceneNodes::UID* mParentIDs;
    static SceneNodes::UID* mSiblingIDs;
    static SceneNodes::UID* mFirstChildIDs;

    static Math::Transform* mGlobalTransforms;
};

// ---------------------------------------------------------------------------
// The cogwheel scene node.
// The scene node implements the Entity Component System, 
// https://en.wikipedia.org/wiki/Entity_component_system, with entities being 
// managed by their own managers, e.g. the Foo component is managed by the FooManager.
// ---------------------------------------------------------------------------
class SceneNode final {
public:
    // -----------------------------------------------------------------------
    // Constructors and destructors
    // -----------------------------------------------------------------------
    SceneNode() : mID(SceneNodes::UID::InvalidUID()) { }
    SceneNode(SceneNodes::UID id) : mID(id) { }
    ~SceneNode() { }

    inline const SceneNodes::UID getID() const { return mID; }

    inline std::string getName() const { return SceneNodes::getName(mID); }
    inline void setName(std::string name) { SceneNodes::setName(mID, name); }

    inline SceneNode getParent() const { return SceneNode(SceneNodes::getParentID(mID)); }
    inline void setParent(SceneNode parent) { SceneNodes::setParent(mID, parent.getID()); }
    inline bool hasChild(SceneNode testedChild) { return SceneNodes::hasChild(mID, testedChild.getID()); }
    std::vector<SceneNode> getChildren() const;

    inline bool exists() const { return SceneNodes::has(mID); }

    inline Math::Transform getLocalTransform() const { return SceneNodes::getLocalTransform(mID); }
    inline void setLocalTransform(Math::Transform transform) { SceneNodes::setLocalTransform(mID, transform); }
    inline Math::Transform getGlobalTransform() const { return SceneNodes::getGlobalTransform(mID); }
    inline void setGlobalTransform(Math::Transform transform) { SceneNodes::setGlobalTransform(mID, transform); }

    template<typename F>
    inline void traverseGraph(F& function) { SceneNodes::traverseGraph<F>(mID, function); }
    template<typename F>
    inline void traverseAllChildren(F& function) { SceneNodes::traverseAllChildren<F>(mID, function); }

    inline bool operator==(SceneNode rhs) const { return mID == rhs.mID; }
    inline bool operator!=(SceneNode rhs) const { return mID != rhs.mID; }

private:
    const SceneNodes::UID mID;
};

// ---------------------------------------------------------------------------
// Temdplate implementations.
// ---------------------------------------------------------------------------

template<typename F>
void SceneNodes::traverseAllChildren(SceneNodes::UID nodeID, F& function) {
    UID node = mFirstChildIDs[nodeID];
    if (node == UID::InvalidUID())
        return;

    do {
        function(node);

        if (mFirstChildIDs[node] != UID::InvalidUID())
            // Visit the next child.
            node = mFirstChildIDs[node];
        else if (mSiblingIDs[node] != UID::InvalidUID())
            // Visit the next sibling.
            node = mSiblingIDs[node];
        else
        {
            // search upwards for next node not visited.
            node = mParentIDs[node];
            while (node != nodeID) {
                UID parentSibling = mSiblingIDs[node];
                if (parentSibling == UID::InvalidUID())
                    node = mParentIDs[node];
                else {
                    node = parentSibling;
                    break;
                }
            }
        }
    } while (node != nodeID);
}

template<typename F>
void SceneNodes::traverseGraph(SceneNodes::UID nodeID, F& function) {
    function(nodeID);
    traverseAllChildren(nodeID, function);
}

} // NS Scene
} // NS Cogwheel

#endif // _COGWHEEL_SCENE_SCENE_NODE_H_