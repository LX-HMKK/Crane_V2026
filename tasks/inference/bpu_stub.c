/* Stub implementations of the libDNN symbols used by bpu_backend.cpp.
 * Linked only on non-RDK hosts (no /usr/lib/hbbpu/libdnn.so) so the full
 * binary builds everywhere; constructing a BpuBackend with these stubs
 * fails cleanly at hbDNNInitializeFromFiles.
 */
#include <stdio.h>
#include <stdlib.h>

#include <dnn/hb_dnn.h>

static int32_t stub_fail(const char* fn) {
  fprintf(stderr,
          "[bpu-stub] %s called on a host without libDNN — BPU backend is "
          "only available on RDK X5\n",
          fn);
  return -1;
}

char const* hbDNNGetVersion() { return "stub-0.0.0"; }

int32_t hbDNNInitializeFromFiles(hbPackedDNNHandle_t* packedDNNHandle,
                                 char const** modelFileNames,
                                 int32_t modelFileCount) {
  (void)packedDNNHandle;
  (void)modelFileNames;
  (void)modelFileCount;
  return stub_fail(__func__);
}

int32_t hbDNNInitializeFromDDR(hbPackedDNNHandle_t* packedDNNHandle,
                               const void** modelData, int32_t* modelDataLengths,
                               int32_t modelDataCount) {
  (void)packedDNNHandle;
  (void)modelData;
  (void)modelDataLengths;
  (void)modelDataCount;
  return stub_fail(__func__);
}

int32_t hbDNNRelease(hbPackedDNNHandle_t packedDNNHandle) {
  (void)packedDNNHandle;
  return 0;
}

int32_t hbDNNGetModelNameList(char const*** modelNameList,
                              int32_t* modelNameCount,
                              hbPackedDNNHandle_t packedDNNHandle) {
  (void)modelNameList;
  (void)modelNameCount;
  (void)packedDNNHandle;
  return stub_fail(__func__);
}

int32_t hbDNNGetModelHandle(hbDNNHandle_t* dnnHandle,
                            hbPackedDNNHandle_t packedDNNHandle,
                            char const* modelName) {
  (void)dnnHandle;
  (void)packedDNNHandle;
  (void)modelName;
  return stub_fail(__func__);
}

int32_t hbDNNGetInputCount(int32_t* inputCount, hbDNNHandle_t dnnHandle) {
  (void)inputCount;
  (void)dnnHandle;
  return stub_fail(__func__);
}

int32_t hbDNNGetInputName(char const** name, hbDNNHandle_t dnnHandle,
                          int32_t inputIndex) {
  (void)name;
  (void)dnnHandle;
  (void)inputIndex;
  return stub_fail(__func__);
}

int32_t hbDNNGetInputTensorProperties(hbDNNTensorProperties* properties,
                                      hbDNNHandle_t dnnHandle,
                                      int32_t inputIndex) {
  (void)properties;
  (void)dnnHandle;
  (void)inputIndex;
  return stub_fail(__func__);
}

int32_t hbDNNGetOutputCount(int32_t* outputCount, hbDNNHandle_t dnnHandle) {
  (void)outputCount;
  (void)dnnHandle;
  return stub_fail(__func__);
}

int32_t hbDNNGetOutputName(char const** name, hbDNNHandle_t dnnHandle,
                           int32_t outputIndex) {
  (void)name;
  (void)dnnHandle;
  (void)outputIndex;
  return stub_fail(__func__);
}

int32_t hbDNNGetOutputTensorProperties(hbDNNTensorProperties* properties,
                                       hbDNNHandle_t dnnHandle,
                                       int32_t outputIndex) {
  (void)properties;
  (void)dnnHandle;
  (void)outputIndex;
  return stub_fail(__func__);
}

int32_t hbDNNInfer(hbDNNTaskHandle_t* taskHandle, hbDNNTensor** output,
                   hbDNNTensor const* input, hbDNNHandle_t dnnHandle,
                   hbDNNInferCtrlParam* inferCtrlParam) {
  (void)taskHandle;
  (void)output;
  (void)input;
  (void)dnnHandle;
  (void)inferCtrlParam;
  return stub_fail(__func__);
}

int32_t hbDNNRoiInfer(hbDNNTaskHandle_t* taskHandle, hbDNNTensor** output,
                      hbDNNTensor const* input, hbDNNRoi* rois,
                      int32_t roiCount, hbDNNHandle_t dnnHandle,
                      hbDNNInferCtrlParam* inferCtrlParam) {
  (void)taskHandle;
  (void)output;
  (void)input;
  (void)rois;
  (void)roiCount;
  (void)dnnHandle;
  (void)inferCtrlParam;
  return stub_fail(__func__);
}

int32_t hbDNNResize(hbDNNTaskHandle_t* taskHandle, hbDNNTensor* output,
                    hbDNNTensor const* input, hbDNNRoi const* roi,
                    hbDNNResizeCtrlParam* resizeCtrlParam) {
  (void)taskHandle;
  (void)output;
  (void)input;
  (void)roi;
  (void)resizeCtrlParam;
  return stub_fail(__func__);
}

int32_t hbDNNSetTaskDoneCb(hbDNNTaskHandle_t taskHandle, hbDNNTaskDoneCb cb,
                           void* userdata) {
  (void)taskHandle;
  (void)cb;
  (void)userdata;
  return stub_fail(__func__);
}

int32_t hbDNNWaitTaskDone(hbDNNTaskHandle_t taskHandle, int32_t timeout) {
  (void)taskHandle;
  (void)timeout;
  return stub_fail(__func__);
}

int32_t hbDNNReleaseTask(hbDNNTaskHandle_t taskHandle) {
  (void)taskHandle;
  return 0;
}

int32_t hbSysAllocMem(hbSysMem* mem, uint32_t size) {
  mem->virAddr = malloc(size);
  mem->phyAddr = 0;
  mem->memSize = mem->virAddr ? size : 0;
  return mem->virAddr ? 0 : HB_SYS_OUT_OF_MEMORY;
}

int32_t hbSysAllocCachedMem(hbSysMem* mem, uint32_t size) {
  return hbSysAllocMem(mem, size);
}

int32_t hbSysFlushMem(hbSysMem* mem, int32_t flag) {
  (void)mem;
  (void)flag;
  return 0;
}

int32_t hbSysWriteMem(hbSysMem* dest, char* src, uint32_t size) {
  (void)dest;
  (void)src;
  (void)size;
  return stub_fail(__func__);
}

int32_t hbSysReadMem(char* dest, hbSysMem* src, uint32_t size) {
  (void)dest;
  (void)src;
  (void)size;
  return stub_fail(__func__);
}

int32_t hbSysFreeMem(hbSysMem* mem) {
  if (mem && mem->virAddr) free(mem->virAddr);
  if (mem) mem->virAddr = NULL;
  return 0;
}

int32_t hbSysRegisterMem(hbSysMem* mem) {
  (void)mem;
  return 0;
}

int32_t hbSysUnregisterMem(hbSysMem* mem) {
  (void)mem;
  return 0;
}
