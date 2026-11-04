#ifndef CUDA_CHECKPOINT_H
#define CUDA_CHECKPOINT_H

#include <stddef.h>

/*
 * Independently authored ABI declarations for the CUDA checkpoint Driver API,
 * based on NVIDIA's public API documentation:
 * https://docs.nvidia.com/cuda/archive/13.0.0/cuda-driver-api/group__CUDA__CHECKPOINT.html
 * Keep these definitions local so the plugin can build without CUDA toolkit
 * headers and resolve the functions from libcuda.so.1 at runtime.
 *
 * This header describes the currently supported CUDA checkpoint ABI. Reserved
 * fields are part of the ABI and must remain zero. A future CUDA release that
 * assigns a meaning to one of them requires an explicit driver-version check
 * and updated declarations after its structure layout has been verified.
 */

typedef int CUresult;

#define CUDA_SUCCESS		 0
#define CUDA_ERROR_INVALID_VALUE 1
#define CUDA_ERROR_NOT_INITIALIZED 3

/* The direct backend uses the CUDA 13.0 checkpoint ABI shipped by r580+. */
#define CUDA_DIRECT_MIN_DRIVER_API_VERSION 13000

typedef enum {
	CU_PROCESS_STATE_RUNNING = 0,
	CU_PROCESS_STATE_LOCKED,
	CU_PROCESS_STATE_CHECKPOINTED,
	CU_PROCESS_STATE_FAILED,
} CUprocessState;

typedef struct {
	unsigned int timeoutMs;
	unsigned int reserved0;
	unsigned long long reserved1[7];
} CUcheckpointLockArgs;

/*
 * Custom storage (CUDA 13.4, r615+): a checkpoint maps the process's GPU
 * memory into the caller's contexts for it to copy out, a restore maps fresh
 * memory for it to copy in, and cuCheckpointOperationComplete() ends either.
 * https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__CHECKPOINT.html
 */
typedef unsigned long long CUdeviceptr;
typedef struct CUstream_st *CUstream;
typedef struct CUctx_st *CUcontext;
typedef int CUdevice;
typedef struct CUIcheckpointOperation_st *CUcheckpointOperationHandle;

typedef struct {
	unsigned char bytes[16];
} CUuuid;

typedef struct {
	CUdeviceptr devPtr;
	size_t size;
	CUstream stream;
} CUcheckpointCustomStoragePerDeviceData;

typedef struct {
	CUcheckpointOperationHandle handle;
	CUcheckpointCustomStoragePerDeviceData *perDeviceData;
	unsigned int deviceCount;
} CUcheckpointCustomStorageInfo;

typedef struct {
	/* NULL: the GPU memory goes to the process's host memory (CUDA 13.0). */
	CUcheckpointCustomStorageInfo **customStorageInfo_out;
	char reserved[64 - sizeof(CUcheckpointCustomStorageInfo **)];
} CUcheckpointCheckpointArgs;

typedef struct {
	unsigned char oldUuid[16];
	unsigned char newUuid[16];
} CUcheckpointGpuPair;

typedef struct {
	CUcheckpointGpuPair *gpuPairs;
	unsigned int gpuPairsCount;
	unsigned int padding0;
	/* NULL: the GPU memory comes from the process's host memory (CUDA 13.0). */
	CUcheckpointCustomStorageInfo **customStorageInfo_out;
	char reserved[64 - sizeof(CUcheckpointGpuPair *) - 2 * sizeof(unsigned int) -
		      sizeof(CUcheckpointCustomStorageInfo **)];
} CUcheckpointRestoreArgs;

typedef struct {
	unsigned long long reserved[8];
} CUcheckpointUnlockArgs;

/* Check the fixed layouts passed to libcuda.so.1. */
_Static_assert(sizeof(CUcheckpointGpuPair) == 32,
	       "CUcheckpointGpuPair must be 32 bytes");
_Static_assert(offsetof(CUcheckpointGpuPair, oldUuid) == 0,
	       "CUcheckpointGpuPair.oldUuid has an unexpected offset");
_Static_assert(offsetof(CUcheckpointGpuPair, newUuid) == 16,
	       "CUcheckpointGpuPair.newUuid has an unexpected offset");

_Static_assert(sizeof(CUcheckpointLockArgs) == 64,
	       "CUcheckpointLockArgs must be 64 bytes");
_Static_assert(sizeof(CUcheckpointCheckpointArgs) == 64,
	       "CUcheckpointCheckpointArgs must be 64 bytes");
_Static_assert(sizeof(CUcheckpointRestoreArgs) == 64,
	       "CUcheckpointRestoreArgs must be 64 bytes");
_Static_assert(sizeof(CUcheckpointUnlockArgs) == 64,
	       "CUcheckpointUnlockArgs must be 64 bytes");

_Static_assert(offsetof(CUcheckpointLockArgs, timeoutMs) == 0,
	       "CUcheckpointLockArgs.timeoutMs has an unexpected offset");
_Static_assert(offsetof(CUcheckpointLockArgs, reserved0) == 4,
	       "CUcheckpointLockArgs.reserved0 has an unexpected offset");
_Static_assert(offsetof(CUcheckpointRestoreArgs, gpuPairs) == 0,
	       "CUcheckpointRestoreArgs.gpuPairs has an unexpected offset");
_Static_assert(offsetof(CUcheckpointRestoreArgs, gpuPairsCount) == sizeof(void *),
	       "CUcheckpointRestoreArgs.gpuPairsCount has an unexpected offset");
_Static_assert(offsetof(CUcheckpointCheckpointArgs, customStorageInfo_out) == 0,
	       "CUcheckpointCheckpointArgs.customStorageInfo_out has an unexpected offset");
_Static_assert(offsetof(CUcheckpointRestoreArgs, customStorageInfo_out) == 16,
	       "CUcheckpointRestoreArgs.customStorageInfo_out has an unexpected offset");
_Static_assert(sizeof(CUuuid) == 16, "CUuuid must be 16 bytes");
_Static_assert(sizeof(CUcheckpointCustomStoragePerDeviceData) == 24,
	       "CUcheckpointCustomStoragePerDeviceData must be 24 bytes");
_Static_assert(offsetof(CUcheckpointCustomStorageInfo, deviceCount) == 16,
	       "CUcheckpointCustomStorageInfo.deviceCount has an unexpected offset");

#endif /* CUDA_CHECKPOINT_H */
