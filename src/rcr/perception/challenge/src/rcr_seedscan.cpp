// Image-estimate query index for the PUBLIC Mulberry32 generator. No labels.
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <omp.h>
struct Node {int scene,next;};
static inline double random_value(uint32_t &state){
 state+=0x6d2b79f5u;uint32_t v=state;
 v=(v^(v>>15))*(v|1u);v^=v+(v^(v>>7))*(v|61u);
 return double(v^(v>>14))/4294967296.0;
}
int main(int argc,char**argv){
 if(argc!=3)return 2;
 uint64_t first=strtoull(argv[1],nullptr,10),end=strtoull(argv[2],nullptr,10);
 constexpr int W=804,H=1404;
 std::vector<int> grid(W*H*9,-1);std::vector<Node> nodes;
 int scene,color,yaw;double x,y;
 while(scanf("%d %lf %lf %d %d",&scene,&x,&y,&color,&yaw)==5){
  int ix=int(std::floor((x+.402)*1000)),iy=int(std::floor((y+.002)*1000));
  for(int dy=-2;dy<=2;dy++)for(int dx=-2;dx<=2;dx++){
   int xx=ix+dx,yy=iy+dy;if(xx<0||xx>=W||yy<0||yy>=H)continue;
   int cell=((color*3+yaw)*H+yy)*W+xx;nodes.push_back({scene,grid[cell]});grid[cell]=int(nodes.size()-1);
  }
 }
 #pragma omp parallel
 {
 std::vector<std::pair<int,uint32_t>> found;
 #pragma omp for schedule(static)
 for(uint64_t seed=first;seed<end;seed++){
  uint32_t state=uint32_t(seed)^0x9e3779b9u;
  int seen[24],seen_attempt[24],nseen=0;
  for(int attempt=0;attempt<6;attempt++){
   int c=int(random_value(state)*3);random_value(state);
   double u=random_value(state),v=random_value(state);if(u+v>1){u=1-u;v=1-v;}
   int a=int(random_value(state)*3);
   const std::array<std::array<double,2>,2> p={{{-.4+.8*(u+v),1.4*v},{-.4+.8*u,1.4*(u+v)}}};
   for(auto point:p){int xx=int(std::floor((point[0]+.402)*1000)),yy=int(std::floor((point[1]+.002)*1000));
    if(xx<0||xx>=W||yy<0||yy>=H)continue;
    for(int node=grid[((c*3+a)*H+yy)*W+xx];node>=0;node=nodes[node].next){
     int scene_id=nodes[node].scene;int idx=0;while(idx<nseen&&seen[idx]!=scene_id)idx++;
     if(idx<nseen){if(seen_attempt[idx]>=0&&seen_attempt[idx]!=attempt){found.emplace_back(scene_id,uint32_t(seed));seen_attempt[idx]=-1;}}
     else if(nseen<24){seen[nseen]=scene_id;seen_attempt[nseen++]=attempt;}
    }
   }
  }
 }
 #pragma omp critical
 for(auto item:found)printf("%d %u\n",item.first,item.second);
 }
}
