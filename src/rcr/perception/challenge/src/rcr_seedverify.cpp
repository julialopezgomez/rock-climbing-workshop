// Conservative positional prefilter for the public native generator.
// Every candidate satisfying the original six-pose consensus must pass here.
// Exact Python geometry/rotation/JSON/RGB verification remains authoritative.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

struct V {double x,y,z;};
V operator+(V a,V b){return {a.x+b.x,a.y+b.y,a.z+b.z};}
V operator-(V a,V b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
V operator*(V a,double b){return {a.x*b,a.y*b,a.z*b};}
V Cross(V a,V b){return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
double Norm2(V a){return a.x*a.x+a.y*a.y+a.z*a.z;}
struct Surface {V a,b,c;double area;};
struct Observation {V point;int color;};
struct Candidate {int scene;uint32_t seed;};
double Random(uint32_t &state){
 state+=0x6d2b79f5u;uint32_t v=state;
 v=(v^(v>>15))*(v|1u);v^=v+(v^(v>>7))*(v|61u);
 return double(v^(v>>14))/4294967296.;
}

bool Generate(uint32_t seed,int nvol,int count,std::vector<Observation> &holds){
 std::vector<Surface> surfaces={{{-.4,0,0},{.4,0,0},{.4,1.4,0},.56},
                                {{-.4,0,0},{.4,1.4,0},{-.4,1.4,0},.56}};
 uint32_t state=seed^0xa511e9b3u;
 for(int i=0;i<nvol;++i){
  double w=std::min(.2,.8*(.1+Random(state)*.15)),h=.08;
  double d=std::min(.2,.8*(.05+Random(state)*.2));
  double x=-.4+w/2+Random(state)*(.8-w),y=h/2+Random(state)*(1.4-h);
  V v[4]={{x-w/2,y-h/2,0},{x+w/2,y-h/2,0},{x,y+h/2,0},{x,y,d}};
  int triangles[3][3]={{0,1,3},{1,2,3},{2,0,3}};
  for(auto &tri:triangles){
   V a=v[tri[0]],b=v[tri[1]],c=v[tri[2]],n=Cross(b-a,c-a);
   double length=std::sqrt(Norm2(n));
   if(std::abs(n.z/length-.25)<1e-10)return false;
   if(n.z/length>.25)surfaces.push_back({a,b,c,length/2});
  }
 }
 double total=0;for(auto s:surfaces)total+=s.area;
 state=seed^0x9e3779b9u;holds.clear();
 for(int i=0;i<count;++i){
  bool placed=false;
  for(int attempt=0;attempt<1000;++attempt){
   int color=int(Random(state)*3);double choice=Random(state)*total;
   Surface selected=surfaces.back();
   for(auto s:surfaces){
    choice-=s.area;
    if(std::abs(choice)<1e-10)return false;
    if(choice<=0){selected=s;break;}
   }
   double u=Random(state),v=Random(state);if(u+v>1){u=1-u;v=1-v;}
   V point=selected.a+(selected.b-selected.a)*u+(selected.c-selected.a)*v;
   Random(state); // yaw is not a positional condition, but consumes the RNG
   bool collision=false;
   for(auto old:holds){
    double distance2=Norm2(point-old.point);
    if(std::abs(distance2-.01)<1e-10)return false;
    if(distance2<.01){collision=true;break;}
   }
   if(collision)continue;
   holds.push_back({point,color});placed=true;break;
  }
  if(!placed)return false;
 }
 return true;
}

bool Augment(int index,const std::vector<std::vector<int>> &edges,std::vector<int> &matched,std::vector<char> &seen){
 for(int j:edges[index]){
  if(seen[j])continue;seen[j]=1;
  if(matched[j]<0||Augment(matched[j],edges,matched,seen)){matched[j]=index;return true;}
 }
 return false;
}

bool Necessary(const std::vector<Observation> &observed,const std::vector<Observation> &generated){
 // Rounded native coordinates differ from unrounded points by <=0.87um.
 // The extra2um also protects floating-point implementation roundoff.
 constexpr double limit=.003002;
 std::vector<std::vector<int>> edges(observed.size());
 for(size_t i=0;i<observed.size();++i)for(size_t j=0;j<generated.size();++j)
  if(observed[i].color==generated[j].color && Norm2(observed[i].point-generated[j].point)<=limit*limit)
   edges[i].push_back(int(j));
 std::vector<int> matched(generated.size(),-1);int count=0;
 for(size_t i=0;i<observed.size();++i){
  std::vector<char> seen(generated.size(),0);
  if(Augment(int(i),edges,matched,seen)&&++count>=6)return true;
 }
 return false;
}

int main(){
 int scenes,nobs,ncandidates,hold_count,maxvol;
 if(scanf("%d %d %d %d %d",&scenes,&nobs,&ncandidates,&hold_count,&maxvol)!=5)return 2;
 if(scenes<1||scenes>10000||nobs<0||ncandidates<0||hold_count<0||hold_count>50||maxvol<0||maxvol>10)return 2;
 std::vector<std::vector<Observation>> observations(scenes);
 for(int i=0;i<nobs;++i){
  int scene,color;V p;
  if(scanf("%d %lf %lf %lf %d",&scene,&p.x,&p.y,&p.z,&color)!=5||scene<0||scene>=scenes||color<0||color>2||!std::isfinite(Norm2(p)))return 2;
  observations[scene].push_back({p,color});
 }
 std::vector<Candidate> candidates(ncandidates);std::vector<char> keep(ncandidates,0);
 for(auto &c:candidates)if(scanf("%d %u",&c.scene,&c.seed)!=2||c.scene<0||c.scene>=scenes)return 2;
 #pragma omp parallel for schedule(dynamic,64)
 for(int i=0;i<ncandidates;++i){
  auto c=candidates[i];std::vector<Observation> generated;
  for(int volumes=0;volumes<=maxvol;++volumes)
   if(!Generate(c.seed,volumes,hold_count,generated)||Necessary(observations[c.scene],generated)){keep[i]=1;break;}
 }
 for(int i=0;i<ncandidates;++i)if(keep[i])printf("%d %u\n",candidates[i].scene,candidates[i].seed);
 return 0;
}
