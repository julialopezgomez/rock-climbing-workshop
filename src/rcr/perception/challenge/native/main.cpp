// Encoder-conditioned M3T adapter. Six root DOFs; all measured links are fixed.
#include <m3t/body.h>
#include <m3t/link.h>
#include <m3t/optimizer.h>
#include <m3t/region_model.h>
#include <m3t/region_modality.h>
#include <m3t/basic_depth_renderer.h>
#include <m3t/renderer_geometry.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

using Pose = m3t::Transform3fA;

Pose ReadPose(const cv::FileNode &node) {
  if (node.size() != 4) throw std::runtime_error("Expected 4x4 pose");
  Pose result = Pose::Identity();
  for (int r=0; r<4; ++r) {
    if (node[r].size() != 4) throw std::runtime_error("Expected 4 pose columns");
    for (int c=0; c<4; ++c) result.matrix()(r,c) = float(double(node[r][c]));
  }
  if (!result.matrix().allFinite() ||
      (result.linear().transpose()*result.linear()-Eigen::Matrix3f::Identity()).norm() > 1e-4 ||
      result.linear().determinant() < .9999f)
    throw std::runtime_error("Invalid rigid transform");
  return result;
}

void Check(bool ok, const std::string &what) {
  if (!ok) throw std::runtime_error("M3T failed: " + what);
}

class PacketCamera : public m3t::ColorCamera {
 public:
  PacketCamera(const std::string &name, const cv::FileNode &node) : ColorCamera(name) {
    auto k = node["K"];
    intrinsics_ = {float(double(k[0][0])), float(double(k[1][1])),
                   float(double(k[0][2])), float(double(k[1][2])),
                   int(node["width"]), int(node["height"])};
  }
  bool Load(const cv::FileNode &node) {
    auto k=node["K"];
    if(int(node["width"])!=intrinsics_.width || int(node["height"])!=intrinsics_.height ||
       std::abs(double(k[0][0])-intrinsics_.fu)>1e-3 || std::abs(double(k[1][1])-intrinsics_.fv)>1e-3 ||
       std::abs(double(k[0][2])-intrinsics_.ppu)>1e-3 || std::abs(double(k[1][2])-intrinsics_.ppv)>1e-3)
      throw std::runtime_error("Camera intrinsics changed; initialize a new tracker");
    if(!node["rgb_bgr_raw"].empty()) {
      std::ifstream stream{std::string(node["rgb_bgr_raw"]),std::ios::binary|std::ios::ate};
      size_t size=size_t(intrinsics_.width)*intrinsics_.height*3;
      if(!stream || stream.tellg()!=std::streampos(size)) return false;
      image_.create(intrinsics_.height,intrinsics_.width,CV_8UC3);
      stream.seekg(0);stream.read(reinterpret_cast<char*>(image_.data),size);
      if(!stream) return false;
    } else image_ = cv::imread(std::string(node["rgb"]));
    set_world2camera_pose(ReadPose(node["T_camera_wall"]));
    return !image_.empty() && image_.cols == intrinsics_.width && image_.rows == intrinsics_.height;
  }
  bool SetUp() override { set_up_ = !image_.empty(); return set_up_; }
  bool UpdateImage(bool) override { return set_up_; }
};

void WritePose(std::ostream &out, const Pose &pose) {
  out << '[';
  for (int r=0;r<4;++r) {
    if (r) out << ',';
    out << '[';
    for (int c=0;c<4;++c) { if(c) out << ','; out << pose.matrix()(r,c); }
    out << ']';
  }
  out << ']';
}

int main(int argc, char **argv) {
 try {
  if(argc==2 && std::string(argv[1])=="--gl-probe") {
    Check(glfwInit(),"hardware EGL initialization");
    auto window=glfwCreateWindow(64,64,"probe",nullptr,nullptr);
    Check(window!=nullptr,"hardware context creation");glfwMakeContextCurrent(window);
    glewExperimental=true;Check(glewInit()==GLEW_OK,"GLEW core GL initialization");
    std::cout << "Vendor: " << glGetString(GL_VENDOR) << "\nRenderer: " << glGetString(GL_RENDERER)
              << "\nVersion: " << glGetString(GL_VERSION) << '\n';
    glfwDestroyWindow(window);return 0;
  }
  if (argc < 4) throw std::runtime_error("Usage: microban_m3t config.json output.jsonl --geometry-only|--gpu-confirmed");
  bool geometry_only = std::string(argv[3]) == "--geometry-only";
  if (!geometry_only && std::string(argv[3]) != "--gpu-confirmed")
    throw std::runtime_error("GPU resource hold: no renderer is initialized without confirmation");
  cv::FileStorage config(argv[1], cv::FileStorage::READ);
  if (!config.isOpened()) throw std::runtime_error("Cannot read config");
  auto geometry = config["links"], frames = config["frames"];
  if (geometry.empty() || frames.empty()) throw std::runtime_error("Empty geometry or input sequence");
  bool independent = int(config["independent"]) != 0;
  bool stream_mode = argc>4 && std::string(argv[4])=="--stream";
  int correspondence_iterations=config["correspondence_iterations"].empty()?7:int(config["correspondence_iterations"]);
  int update_iterations=config["update_iterations"].empty()?2:int(config["update_iterations"]);
  if(correspondence_iterations<1||correspondence_iterations>30||update_iterations<1||update_iterations>8)
    throw std::runtime_error("Invalid M3T iteration budget");
  std::vector<std::shared_ptr<m3t::Body>> bodies;
  std::vector<std::shared_ptr<m3t::Link>> links;
  std::vector<std::shared_ptr<m3t::RegionModel>> models;
  std::vector<std::shared_ptr<m3t::RegionModality>> modalities;
  std::vector<std::shared_ptr<PacketCamera>> cameras;
  std::vector<std::shared_ptr<m3t::FocusedBasicDepthRenderer>> depth_renderers;
  for (const auto &item : geometry) {
    auto name = std::string(item["name"]);
    auto body = std::make_shared<m3t::Body>(name, std::string(item["geometry"]), 1.f, true, false, Pose::Identity());
    Check(body->SetUp(), "load visual mesh " + name);
    bodies.push_back(body);
    auto link = std::make_shared<m3t::Link>(name, body);
    if (links.size()) {
      link->set_free_directions({false,false,false,false,false,false});
      Check(links[0]->AddChildLink(link), "attach fixed encoder link");
    }
    links.push_back(link);
  }
  if (!geometry_only) {
    auto render_geometry = std::make_shared<m3t::RendererGeometry>("microban_geometry");
    for (auto &body : bodies) Check(render_geometry->AddBody(body), "renderer body");
    Check(render_geometry->SetUp(), "renderer geometry");
    int eye=0;
    for (const auto &item : frames[0]["cameras"]) {
      auto camera=std::make_shared<PacketCamera>("camera_"+std::to_string(eye),item);
      Check(camera->Load(item), "load first RGB");
      Check(camera->SetUp(), "camera setup");
      cameras.push_back(camera);
      auto depth=std::make_shared<m3t::FocusedBasicDepthRenderer>("occlusion_"+std::to_string(eye),render_geometry,camera,400);
      for (auto &body:bodies) Check(depth->AddReferencedBody(body),"occluder reference");
      Check(depth->SetUp(), "modelled occlusion renderer");
      depth_renderers.push_back(depth);
      ++eye;
    }
    auto cache = std::filesystem::path(std::string(config["model_cache"]));
    std::filesystem::create_directories(cache);
    for (size_t i=0;i<bodies.size();++i) {
      auto model=std::make_shared<m3t::RegionModel>(bodies[i]->name()+"_region",bodies[i],cache/(bodies[i]->name()+".bin"),.4f,2,200,.02f,.002f,false,1000);
      Check(model->SetUp(), "sparse viewpoint model " + bodies[i]->name());
      models.push_back(model);
      for (size_t eye=0;eye<cameras.size();++eye) {
        auto modality=std::make_shared<m3t::RegionModality>(bodies[i]->name()+"_"+std::to_string(eye),bodies[i],cameras[eye],model);
        modality->set_n_lines_max(150);
        modality->set_scales({5,2,2,1});
        modality->set_standard_deviations({7.f,4.f,2.f,1.5f});
        modality->ModelOcclusions(depth_renderers[eye]);
        Check(modality->SetUp(), "region modality");
        Check(links[i]->AddModality(modality), "attach region modality");
        modalities.push_back(modality);
      }
    }
  }
  for (auto &link:links) Check(link->SetUp(), "kinematic link");
  auto optimizer=std::make_shared<m3t::Optimizer>("six_dof_root",links[0]);
  Check(optimizer->SetUp(), "optimizer");
  std::ofstream output(argv[2]);
  if(!output) throw std::runtime_error("Cannot create output");
  output << std::setprecision(10);
  int index=0;
  auto process_frame = [&](const cv::FileNode &frame,bool force_reset) {
    auto started=std::chrono::steady_clock::now();
    std::vector<Pose> offsets;
    for (const auto &offset:frame["T_trunk_meshes"]) offsets.push_back(ReadPose(offset));
    if(offsets.size()!=links.size()) throw std::runtime_error("Encoder transform count differs from links");
    auto initial=ReadPose(frame["T_wall_trunk_initial"]);
    for (size_t i=1;i<links.size();++i)
      links[i]->set_joint2parent_pose(offsets[0].inverse()*offsets[i]);
    bool reset=independent || force_reset || !index;
    if (geometry_only || reset) bodies[0]->set_body2world_pose(initial*offsets[0]);
    Check(optimizer->CalculateConsistentPoses(), "apply known encoders");
    float error=0.f;
    if(geometry_only) {
      for(size_t i=0;i<bodies.size();++i)
        error=std::max(error,(bodies[i]->body2world_pose().matrix()-(initial*offsets[i]).matrix()).cwiseAbs().maxCoeff());
      if(error>1e-5f) throw std::runtime_error("Fixed-joint pose chain is inconsistent");
    } else {
      for(size_t eye=0;eye<cameras.size();++eye) Check(cameras[eye]->Load(frame["cameras"][int(eye)]),"RGB update");
      if(reset) {
        for(auto &depth:depth_renderers) Check(depth->StartRendering(),"initial occlusion render");
        for(auto &modality:modalities) Check(modality->StartModality(index,0),"initialize image histograms");
      }
      for(int corr=0;corr<correspondence_iterations;++corr) {
        for(auto &depth:depth_renderers) Check(depth->StartRendering(),"occlusion render");
        for(auto &modality:modalities) Check(modality->CalculateCorrespondences(index,corr),"RGB correspondences");
        for(int update=0;update<update_iterations;++update) {
          for(auto &modality:modalities) Check(modality->CalculateGradientAndHessian(index,corr,update),"image gradient");
          Check(optimizer->CalculateOptimization(index,corr,update),"root update");
        }
      }
      for(auto &depth:depth_renderers) Check(depth->StartRendering(),"result occlusion render");
      for(auto &modality:modalities) Check(modality->CalculateResults(index),"histogram update");
    }
    auto pose=bodies[0]->body2world_pose()*offsets[0].inverse();
    double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
    std::ostringstream row;row << std::setprecision(10);
    row << "{\"case\":\"" << std::string(frame["id"]) << "\",\"geometry_only\":" << (geometry_only?"true":"false")
           << ",\"seconds\":" << seconds << ",\"fk_max_matrix_error\":" << error << ",\"T_wall_trunk\":";
    WritePose(row,pose); row << "}\n";
    output << row.str();output.flush();
    if(stream_mode) std::cout << row.str() << std::flush;
    ++index;
  };
  for (const auto &frame:frames) process_frame(frame,false);
  if(stream_mode) {
    std::string line;
    while(std::getline(std::cin,line)) {
      if(line.empty()) continue;
      cv::FileStorage packet(line,cv::FileStorage::READ|cv::FileStorage::MEMORY|cv::FileStorage::FORMAT_JSON);
      if(!packet.isOpened()) throw std::runtime_error("Invalid streamed JSON frame");
      if(!packet["stop"].empty() && int(packet["stop"])) break;
      process_frame(packet["frame"],!packet["reset"].empty() && int(packet["reset"]));
    }
  }
  std::cout << "Completed " << index << " frames, " << bodies.size() << " CAD links, geometry_only=" << geometry_only << '\n';
  return 0;
 } catch(const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
