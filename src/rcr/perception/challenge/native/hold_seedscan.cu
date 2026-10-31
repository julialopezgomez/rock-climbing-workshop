// Four-type public sampler query: color, type, surface, u, v, yaw.
// Input is RGB-fitted wall anchors only. No labels or seed hints.
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

constexpr int MAX_SCENES=128;
struct Node { int scene, next; };
struct Hit { uint32_t scene, seed; };

void Check(cudaError_t status) {
  if(status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

__device__ __forceinline__ double Random(uint32_t &state) {
  state+=0x6d2b79f5u;
  uint32_t v=state;
  v=(v^(v>>15))*(v|1u);
  v^=v+(v^(v>>7))*(v|61u);
  return double(v^(v>>14))/4294967296.0;
}

__global__ void Scan(uint64_t first,uint64_t end,const int *grid,const Node *nodes,
                     Hit *hits,unsigned int *count,unsigned int capacity, int W, int H, double width, double height) {
  uint64_t seed=first+uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(seed>=end) return;
  uint32_t state=uint32_t(seed)^0x9e3779b9u;
  uint32_t seen[4]={0,0,0,0}, twice[4]={0,0,0,0}, emitted[4]={0,0,0,0};
  for(int attempt=0;attempt<12;++attempt) {
    int color=int(Random(state)*3);
    int kind=int(Random(state)*3);
    Random(state);
    double u=Random(state),v=Random(state);
    if(u+v>1) {u=1-u;v=1-v;}
    Random(state);
    double px[2]={-width/2+width*(u+v),-width/2+width*u};
    double py[2]={height*v,height*(u+v)};
    uint32_t current[4]={0,0,0,0};
    for(int triangle=0;triangle<2;++triangle) {
      int x=int(floor((px[triangle]+width/2+.004)*1000));
      int y=int(floor((py[triangle]+.004)*1000));
      if(x<0||x>=W||y<0||y>=H) continue;
      int cell=((color*3+kind)*H+y)*W+x;
      for(int node=grid[cell];node>=0;node=nodes[node].next) {
        unsigned int scene=nodes[node].scene;
        current[scene>>5]|=1u<<(scene&31);
      }
    }
    for(int word=0;word<4;++word) {
      uint32_t fresh=(twice[word]&current[word])&~emitted[word];
      emitted[word]|=fresh;
      twice[word]|=seen[word]&current[word];
      seen[word]|=current[word];
      while(fresh) {
        int bit=__ffs(fresh)-1;
        unsigned int index=atomicAdd(count,1u);
        if(index<capacity) hits[index]={uint32_t(word*32+bit),uint32_t(seed)};
        fresh&=fresh-1;
      }
    }
  }
}

int main(int argc,char **argv) {
 try {
  if(argc!=5) throw std::runtime_error("Usage: rcr-seedscan-cuda first end width height < RGB_queries.txt");
  uint64_t first=std::stoull(argv[1]),end=std::stoull(argv[2]);
  if(first>=end||end>(1ull<<32)) throw std::runtime_error("Invalid unsigned 32-bit interval");
  double width=std::stod(argv[3]),height=std::stod(argv[4]);
  if (!(width>0 && width<=10 && height>0 && height<=10)) throw std::runtime_error("Invalid wall dimensions");
  int W=int(std::ceil((width+.008)*1000)),H=int(std::ceil((height+.008)*1000));
  std::vector<int> grid(W*H*9,-1);
  std::vector<Node> nodes;
  int scene,color,kind;double x,y;
  while(scanf("%d %lf %lf %d %d",&scene,&x,&y,&color,&kind)==5) {
    if(scene<0||scene>=MAX_SCENES||color<0||color>2||kind<0||kind>2||!std::isfinite(x)||!std::isfinite(y))
      throw std::runtime_error("Invalid query or more than 128 scenes in one scan");
    int ix=int(std::floor((x+width/2+.004)*1000)),iy=int(std::floor((y+.004)*1000));
    for(int dy=-3;dy<=3;++dy) for(int dx=-3;dx<=3;++dx) {
      int xx=ix+dx,yy=iy+dy;
      if(xx<0||xx>=W||yy<0||yy>=H) continue;
      int cell=((color*3+kind)*H+yy)*W+xx;
      nodes.push_back({scene,grid[cell]});grid[cell]=int(nodes.size()-1);
    }
  }
  if(nodes.empty()) return 0;
  int *device_grid=nullptr;Node *device_nodes=nullptr;
  Hit *device_hits=nullptr;unsigned int *device_count=nullptr;
  constexpr unsigned int capacity=1000000;
  Check(cudaMalloc(&device_grid,grid.size()*sizeof(int)));
  Check(cudaMalloc(&device_nodes,nodes.size()*sizeof(Node)));
  Check(cudaMalloc(&device_hits,capacity*sizeof(Hit)));
  Check(cudaMalloc(&device_count,sizeof(unsigned int)));
  Check(cudaMemcpy(device_grid,grid.data(),grid.size()*sizeof(int),cudaMemcpyHostToDevice));
  Check(cudaMemcpy(device_nodes,nodes.data(),nodes.size()*sizeof(Node),cudaMemcpyHostToDevice));
  constexpr uint64_t chunk=1ull<<25;
  std::vector<Hit> found;
  for(uint64_t start=first;start<end;start+=chunk) {
    uint64_t stop=std::min(end,start+chunk);
    Check(cudaMemset(device_count,0,sizeof(unsigned int)));
    Scan<<<unsigned((stop-start+255)/256),256>>>(start,stop,device_grid,device_nodes,device_hits,device_count,capacity,W,H,width,height);
    Check(cudaGetLastError());
    unsigned int count=0;
    Check(cudaMemcpy(&count,device_count,sizeof(count),cudaMemcpyDeviceToHost));
    if(count>capacity) throw std::runtime_error("Candidate buffer overflow; scan a smaller interval");
    auto offset=found.size();found.resize(offset+count);
    if(count) Check(cudaMemcpy(found.data()+offset,device_hits,count*sizeof(Hit),cudaMemcpyDeviceToHost));
  }
  std::sort(found.begin(),found.end(),[](const Hit &a,const Hit &b) {
    return a.scene<b.scene||(a.scene==b.scene&&a.seed<b.seed);
  });
  for(auto h:found) printf("%u %u\n",h.scene,h.seed);
  Check(cudaFree(device_grid));Check(cudaFree(device_nodes));
  Check(cudaFree(device_hits));Check(cudaFree(device_count));
  return 0;
 } catch(const std::exception &error) { fprintf(stderr,"%s\n",error.what());return 1; }
}
