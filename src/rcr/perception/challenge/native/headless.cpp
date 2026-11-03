// Headless NVIDIA EGL context for the exact GLFW subset used by M3T.
// The upstream tracker is unchanged. No software renderer is accepted.
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <dlfcn.h>
#include <cstring>
#include <iostream>
#include <mutex>

struct GLFWwindow { EGLSurface surface; EGLContext context; };
namespace {
EGLDisplay display = EGL_NO_DISPLAY;
EGLConfig config;
std::once_flag initialize;
thread_local GLFWwindow *current = nullptr;
}

extern "C" int glfwInit() {
  std::call_once(initialize, [] {
    auto query = reinterpret_cast<PFNEGLQUERYDEVICESEXTPROC>(eglGetProcAddress("eglQueryDevicesEXT"));
    auto get_display = reinterpret_cast<PFNEGLGETPLATFORMDISPLAYEXTPROC>(eglGetProcAddress("eglGetPlatformDisplayEXT"));
    if (!query || !get_display) return;
    EGLDeviceEXT devices[16]; EGLint count=0;
    if (!query(16,devices,&count)) return;
    for (int i=0;i<count;++i) {
      auto candidate=get_display(EGL_PLATFORM_DEVICE_EXT,devices[i],nullptr);
      EGLint major=0,minor=0;
      if (!eglInitialize(candidate,&major,&minor)) continue;
      auto vendor=eglQueryString(candidate,EGL_VENDOR);
      if (!vendor || !std::strstr(vendor,"NVIDIA")) { eglTerminate(candidate); continue; }
      EGLint attributes[]={EGL_SURFACE_TYPE,EGL_PBUFFER_BIT,EGL_RENDERABLE_TYPE,EGL_OPENGL_BIT,
        EGL_RED_SIZE,8,EGL_GREEN_SIZE,8,EGL_BLUE_SIZE,8,EGL_DEPTH_SIZE,24,EGL_NONE};
      EGLint found=0;
      if (!eglChooseConfig(candidate,attributes,&config,1,&found) || !found) continue;
      display=candidate;
      std::cerr << "M3T headless EGL vendor: " << vendor << '\n';
      break;
    }
  });
  return display!=EGL_NO_DISPLAY;
}

extern "C" void glfwWindowHint(int, int) {}
extern "C" void glfwTerminate() {}  // process-global display, reusable by model caches
extern "C" GLFWwindow *glfwCreateWindow(int width,int height,const char *,GLFWmonitor *,GLFWwindow *share) {
  if(!glfwInit() || !eglBindAPI(EGL_OPENGL_API)) return nullptr;
  EGLint surface_attributes[]={EGL_WIDTH,width,EGL_HEIGHT,height,EGL_NONE};
  EGLint context_attributes[]={EGL_CONTEXT_MAJOR_VERSION,3,EGL_CONTEXT_MINOR_VERSION,3,
    EGL_CONTEXT_OPENGL_PROFILE_MASK,EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,EGL_NONE};
  auto surface=eglCreatePbufferSurface(display,config,surface_attributes);
  auto context=eglCreateContext(display,config,share?share->context:EGL_NO_CONTEXT,context_attributes);
  if(surface==EGL_NO_SURFACE || context==EGL_NO_CONTEXT) return nullptr;
  return new GLFWwindow{surface,context};
}
extern "C" void glfwMakeContextCurrent(GLFWwindow *window) {
  if (window) {
    eglBindAPI(EGL_OPENGL_API);
    if(!eglMakeCurrent(display,window->surface,window->surface,window->context)) {
      std::cerr << "eglMakeCurrent failed: " << eglGetError() << '\n'; std::abort();
    }
  } else if (display!=EGL_NO_DISPLAY) eglMakeCurrent(display,EGL_NO_SURFACE,EGL_NO_SURFACE,EGL_NO_CONTEXT);
  current=window;
}
extern "C" GLFWwindow *glfwGetCurrentContext() {return current;}
extern "C" void glfwDestroyWindow(GLFWwindow *window) {
  if(!window) return;
  if(current==window) glfwMakeContextCurrent(nullptr);
  eglDestroyContext(display,window->context);eglDestroySurface(display,window->surface);delete window;
}
extern "C" GLenum glewInit() {
  auto original=reinterpret_cast<GLenum(*)()>(dlsym(RTLD_NEXT,"glewInit"));
  if(!original) return GLEW_ERROR_NO_GL_VERSION;
  auto status=original();
  // GLX is unused with EGL. GLEW has already populated the core GL table when
  // its optional GLX initialization reports that no X display is present.
  auto vendor=reinterpret_cast<const char *>(glGetString(GL_VENDOR));
  if(status==GLEW_ERROR_NO_GLX_DISPLAY && vendor && std::strstr(vendor,"NVIDIA") &&
     glGenVertexArrays && glGenBuffers && glBindFramebuffer && glCreateShader) return GLEW_OK;
  return status;
}
