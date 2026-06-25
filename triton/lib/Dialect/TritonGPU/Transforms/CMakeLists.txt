add_triton_library(TritonGPUTransforms
  CoalesceUtils.cpp
  Utility.cpp

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
  MLIRTransforms
  MLIRTransformUtils
  TritonAnalysis
  TritonIR
  TritonGPUIR
  TritonNvidiaGPUIR
)
