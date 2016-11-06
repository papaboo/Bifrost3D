// DirectX 11 renderer host types.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _DX11RENDERER_RENDERER_TYPES_H_
#define _DX11RENDERER_RENDERER_TYPES_H_

struct ID3D11Buffer;

namespace DX11Renderer {

struct Dx11Mesh {
    unsigned int index_count;
    unsigned int vertex_count;
    ID3D11Buffer* indices; // What is the concrete implementation of this buffer? Do I really need to always use the interface?
    ID3D11Buffer* positions;
};

struct Dx11Model {
    unsigned int mesh_ID; // TODO Consider storing a mesh here, instead of a reference to the mesh.
    unsigned int transform_ID;
    unsigned int material_ID;
};

} // NS DX11Renderer

#endif // _DX11RENDERER_RENDERER_TYPES_H_