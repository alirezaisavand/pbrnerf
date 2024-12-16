#pragma once

#include "gdt/math/vec.h"

using namespace gdt;

struct TriangleMeshSBTData {
  vec3f  color;
  vec3f *vertex;
  vec3i *index;
};

struct TriangleMeshSBTDataBuf {
  float *vertex;
  uint32_t *index;
};

struct RaySBTData {
  vec3f *origin;
  vec3f *dir;
};

struct RaySBTDataBuf {
  float *origin;
  float *dir;
};

struct LaunchParams
{
  struct {
    float *colorBuffer;
    vec2i     size;
  } frame;
  
  struct {
    vec3f position;
    vec3f direction;
    vec3f horizontal;
    vec3f vertical;
  } camera;

  OptixTraversableHandle traversable;
};
