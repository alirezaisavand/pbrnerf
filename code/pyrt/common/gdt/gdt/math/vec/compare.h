#pragma once

namespace gdt {

  // ------------------------------------------------------------------
  // ==
  // ------------------------------------------------------------------

#if __CUDACC__
  template<typename T>
  inline __both__ bool operator==(const vec_t<T,2> &a, const vec_t<T,2> &b)
  { return (a.x==b.x) & (a.y==b.y); }
  
  template<typename T>
  inline __both__ bool operator==(const vec_t<T,3> &a, const vec_t<T,3> &b)
  { return (a.x==b.x) & (a.y==b.y) & (a.z==b.z); }
  
  template<typename T>
  inline __both__ bool operator==(const vec_t<T,4> &a, const vec_t<T,4> &b)
  { return (a.x==b.x) & (a.y==b.y) & (a.z==b.z) & (a.w==b.w); }
#else
  template<typename T>
  inline __both__ bool operator==(const vec_t<T,2> &a, const vec_t<T,2> &b)
  { return a.x==b.x && a.y==b.y; }

  template<typename T>
  inline __both__ bool operator==(const vec_t<T,3> &a, const vec_t<T,3> &b)
  { return a.x==b.x && a.y==b.y && a.z==b.z; }

  template<typename T>
  inline __both__ bool operator==(const vec_t<T,4> &a, const vec_t<T,4> &b)
  { return a.x==b.x && a.y==b.y && a.z==b.z && a.w==b.w; }
#endif
  
  // ------------------------------------------------------------------
  // !=
  // ------------------------------------------------------------------
  
  template<typename T, int N>
  inline __both__ bool operator!=(const vec_t<T,N> &a, const vec_t<T,N> &b)
  { return !(a==b); }
  
} // ::gdt
