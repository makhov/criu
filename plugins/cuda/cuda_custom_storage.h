#ifndef CUDA_CUSTOM_STORAGE_H
#define CUDA_CUSTOM_STORAGE_H

#include <stdbool.h>
#include <sys/types.h>

#include "cuda_device_map.h"

/*
 * GPU memory checkpointed to, and restored from, an image file of its own
 * through the driver's custom storage (CUDA 13.4, r615+), instead of through
 * the process's host memory: the memory goes between the GPU and the file in
 * pieces, and the process's memory never holds it.
 */

/* Whether checkpoints go through custom storage: libcuda has it, images are
 * files (not streamed), and cuda_plugin.storage did not ask for host memory.
 */
bool cuda_custom_storage_enabled(void *libcuda);

/* Whether a custom storage image of pid's GPU memory exists. */
bool cuda_custom_storage_image(pid_t pid);

/* Checkpoint pid, LOCKED, into its image: CHECKPOINTED on success. */
int cuda_custom_storage_checkpoint(void *libcuda, pid_t pid);

/* Restore pid, CHECKPOINTED, from its image onto the GPUs map gives (NULL:
 * the same GPUs): LOCKED on success.
 */
int cuda_custom_storage_restore(void *libcuda, pid_t pid, const struct cuda_device_map *map);

#endif /* CUDA_CUSTOM_STORAGE_H */
