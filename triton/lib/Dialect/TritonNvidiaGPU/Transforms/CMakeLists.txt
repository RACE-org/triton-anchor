add_triton_library(TritonNvidiaGPUTransforms
  TMAUtilities.cpp

  DEPENDS
  TritonTableGen
  TritonGPUTableGen
  TritonGPUCGAAttrIncGen
  TritonGPUAttrDefsIncGen
  TritonGPUTypeInterfacesIncGen
  TritonGPUOpInterfacesIncGen
  TritonNvidiaGPUTableGen
  TritonNvidiaGPUAttrDefsIncGen
  TritonNvidiaGPUOpInterfacesIncGen

  LINK_LIBS PUBLIC
  TritonIR
  TritonGPUIR
  TritonNvidiaGPUIR
)
