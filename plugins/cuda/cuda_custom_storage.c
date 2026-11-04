#include "cuda_custom_storage.h"
#include "cuda_plugin.h"
#include "cuda_wait.h"
#include "criu-plugin.h"
#include "criu-log.h"
#include "cr_options.h"

#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/wait.h>
#include <unistd.h>

#ifdef LOG_PREFIX
#undef LOG_PREFIX
#endif
#define LOG_PREFIX "cuda_plugin: "

/*
 * cuda-mem-<vpid>.img: a header, a table with one entry per GPU, and each
 * GPU's memory at its entry's offset. The vpid is the task's pid in its own
 * pid namespace, which a restore gives it again; a checkpoint writes the
 * image under a temporary name and renames it once it is whole, so an
 * image that exists is complete.
 */
#define CUDA_MEM_IMAGE	   "cuda-mem-%d.img"
#define CUDA_MEM_MAGIC	   "CRCUDAM1"
#define CUDA_MEM_VERSION   1
#define CUDA_MEM_ALIGN	   4096
#define CUDA_MEM_MAX_GPUS  64
/* The pinned buffer the memory goes through. */
#define CUDA_MEM_CHUNK	   (64UL << 20)

struct cuda_mem_header {
	char magic[8];
	uint32_t version;
	uint32_t count;
};

struct cuda_mem_entry {
	unsigned char uuid[16];
	uint64_t size;
	uint64_t offset;
};

struct cuda_storage_api {
	CUresult (*init)(unsigned int flags);
	CUresult (*device_get_count)(int *count);
	CUresult (*device_get)(CUdevice *device, int ordinal);
	CUresult (*device_get_uuid)(CUuuid *uuid, CUdevice device);
	CUresult (*primary_ctx_retain)(CUcontext *ctx, CUdevice device);
	CUresult (*stream_get_ctx)(CUstream stream, CUcontext *ctx);
	CUresult (*ctx_push)(CUcontext ctx);
	CUresult (*ctx_pop)(CUcontext *ctx);
	CUresult (*ctx_get_device)(CUdevice *device);
	CUresult (*mem_host_alloc)(void **ptr, size_t size, unsigned int flags);
	CUresult (*memcpy_dtoh)(void *dst, CUdeviceptr src, size_t size, CUstream stream);
	CUresult (*memcpy_htod)(CUdeviceptr dst, const void *src, size_t size, CUstream stream);
	CUresult (*stream_synchronize)(CUstream stream);
	CUresult (*checkpoint)(int pid, CUcheckpointCheckpointArgs *args);
	CUresult (*restore)(int pid, CUcheckpointRestoreArgs *args);
	CUresult (*complete)(CUcheckpointOperationHandle handle);
};

static struct cuda_storage_api api;
static bool api_loaded;

/* cuda_plugin.storage=host turns custom storage off. */
bool cuda_plugin_host_storage;

static int load_api(void *libcuda)
{
	struct {
		const char *name;
		void **fn;
	} syms[] = {
		{ "cuInit", (void **)&api.init },
		{ "cuDeviceGetCount", (void **)&api.device_get_count },
		{ "cuDeviceGet", (void **)&api.device_get },
		/* The GPU's UUID, not a MIG instance's (cuDeviceGetUuid_v2). */
		{ "cuDeviceGetUuid", (void **)&api.device_get_uuid },
		{ "cuDevicePrimaryCtxRetain", (void **)&api.primary_ctx_retain },
		{ "cuStreamGetCtx", (void **)&api.stream_get_ctx },
		{ "cuCtxPushCurrent_v2", (void **)&api.ctx_push },
		{ "cuCtxPopCurrent_v2", (void **)&api.ctx_pop },
		{ "cuCtxGetDevice", (void **)&api.ctx_get_device },
		{ "cuMemHostAlloc", (void **)&api.mem_host_alloc },
		{ "cuMemcpyDtoHAsync_v2", (void **)&api.memcpy_dtoh },
		{ "cuMemcpyHtoDAsync_v2", (void **)&api.memcpy_htod },
		{ "cuStreamSynchronize", (void **)&api.stream_synchronize },
		{ "cuCheckpointProcessCheckpoint", (void **)&api.checkpoint },
		{ "cuCheckpointProcessRestore", (void **)&api.restore },
		{ "cuCheckpointOperationComplete", (void **)&api.complete },
	};
	size_t i;

	if (api_loaded)
		return 0;
	if (!libcuda)
		return -ENOSYS;
	for (i = 0; i < sizeof(syms) / sizeof(syms[0]); i++) {
		*syms[i].fn = dlsym(libcuda, syms[i].name);
		if (!*syms[i].fn)
			return -ENOSYS;
	}
	api_loaded = true;
	return 0;
}

bool cuda_custom_storage_enabled(void *libcuda)
{
	if (cuda_plugin_host_storage || opts.stream || !libcuda)
		return false;
	return load_api(libcuda) == 0;
}

/* The pid in its own pid namespace: the last NSpid. */
static int task_vpid(pid_t pid)
{
	char path[64], line[256];
	int vpid = -1;
	FILE *f;

	snprintf(path, sizeof(path), "/proc/%d/status", pid);
	f = fopen(path, "re");
	if (!f)
		return -1;
	while (fgets(line, sizeof(line), f)) {
		char *p, *end;

		if (strncmp(line, "NSpid:", 6))
			continue;
		for (p = line + 6;; p = end) {
			long n = strtol(p, &end, 10);

			if (end == p)
				break;
			vpid = (int)n;
		}
		break;
	}
	fclose(f);
	return vpid;
}

static int image_name(pid_t pid, char *name, size_t size)
{
	int vpid = task_vpid(pid);

	if (vpid <= 0) {
		pr_err("Unable to find the namespace pid of %d\n", pid);
		return -1;
	}
	snprintf(name, size, CUDA_MEM_IMAGE, vpid);
	return 0;
}

bool cuda_custom_storage_image(pid_t pid)
{
	char name[64];
	struct stat st;

	if (opts.stream || image_name(pid, name, sizeof(name)))
		return false;
	return fstatat(criu_get_image_dir(), name, &st, 0) == 0;
}

static int full_pwrite(int fd, const void *buf, size_t size, off_t off)
{
	while (size) {
		ssize_t n = pwrite(fd, buf, size, off);

		if (n < 0) {
			if (errno == EINTR)
				continue;
			return -errno;
		}
		buf = (const char *)buf + n;
		size -= n;
		off += n;
	}
	return 0;
}

static int full_pread(int fd, void *buf, size_t size, off_t off)
{
	while (size) {
		ssize_t n = pread(fd, buf, size, off);

		if (n < 0) {
			if (errno == EINTR)
				continue;
			return -errno;
		}
		if (!n)
			return -EIO;
		buf = (char *)buf + n;
		size -= n;
		off += n;
	}
	return 0;
}

static void leave_device(void);

/* Pinned host memory every context can copy through. */
#define CU_MEMHOSTALLOC_PORTABLE 0x01

/*
 * The driver wants the caller to hold the primary contexts of the GPUs. The
 * pinned buffer is allocated here too, before the checkpoint or restore
 * starts: a failure now leaves the task as it was.
 */
static int retain_contexts(void **buf)
{
	CUcontext first = NULL;
	int count, i;

	if (api.init(0) != CUDA_SUCCESS || api.device_get_count(&count) != CUDA_SUCCESS) {
		pr_err("Unable to initialize CUDA for custom storage\n");
		return -1;
	}
	for (i = 0; i < count; i++) {
		CUcontext ctx;
		CUdevice dev;

		if (api.device_get(&dev, i) != CUDA_SUCCESS || api.primary_ctx_retain(&ctx, dev) != CUDA_SUCCESS) {
			pr_err("Unable to retain the primary context of CUDA device %d\n", i);
			return -1;
		}
		if (!first)
			first = ctx;
	}
	if (!first || api.ctx_push(first) != CUDA_SUCCESS) {
		pr_err("No CUDA context to allocate a pinned buffer in\n");
		return -1;
	}
	if (api.mem_host_alloc(buf, CUDA_MEM_CHUNK, CU_MEMHOSTALLOC_PORTABLE) != CUDA_SUCCESS) {
		pr_err("Unable to allocate a pinned buffer for GPU memory\n");
		leave_device();
		return -1;
	}
	leave_device();
	return 0;
}

/* Makes the context of d's stream current and gives the GPU's UUID. */
static int enter_device(const CUcheckpointCustomStoragePerDeviceData *d, CUuuid *uuid)
{
	CUcontext ctx;
	CUdevice dev;

	if (api.stream_get_ctx(d->stream, &ctx) != CUDA_SUCCESS || api.ctx_push(ctx) != CUDA_SUCCESS ||
	    api.ctx_get_device(&dev) != CUDA_SUCCESS || api.device_get_uuid(uuid, dev) != CUDA_SUCCESS) {
		pr_err("Unable to enter the context of a custom storage GPU\n");
		return -1;
	}
	return 0;
}

static void leave_device(void)
{
	CUcontext ctx;

	api.ctx_pop(&ctx);
}

static int copy_out(int fd, const CUcheckpointCustomStoragePerDeviceData *d, void *buf, off_t off)
{
	size_t done, n;
	int ret;

	for (done = 0; done < d->size; done += n) {
		n = d->size - done < CUDA_MEM_CHUNK ? d->size - done : CUDA_MEM_CHUNK;
		if (api.memcpy_dtoh(buf, d->devPtr + done, n, d->stream) != CUDA_SUCCESS ||
		    api.stream_synchronize(d->stream) != CUDA_SUCCESS) {
			pr_err("Unable to copy GPU memory out at %zu\n", done);
			return -1;
		}
		ret = full_pwrite(fd, buf, n, off + done);
		if (ret) {
			pr_err("Unable to write GPU memory: %s\n", strerror(-ret));
			return -1;
		}
	}
	return 0;
}

static int copy_in(int fd, const CUcheckpointCustomStoragePerDeviceData *d, void *buf, off_t off)
{
	size_t done, n;
	int ret;

	for (done = 0; done < d->size; done += n) {
		n = d->size - done < CUDA_MEM_CHUNK ? d->size - done : CUDA_MEM_CHUNK;
		ret = full_pread(fd, buf, n, off + done);
		if (ret) {
			pr_err("Unable to read GPU memory: %s\n", strerror(-ret));
			return -1;
		}
		if (api.memcpy_htod(d->devPtr + done, buf, n, d->stream) != CUDA_SUCCESS ||
		    api.stream_synchronize(d->stream) != CUDA_SUCCESS) {
			pr_err("Unable to copy GPU memory in at %zu\n", done);
			return -1;
		}
	}
	return 0;
}

static uint64_t align_up(uint64_t v)
{
	return (v + CUDA_MEM_ALIGN - 1) & ~(uint64_t)(CUDA_MEM_ALIGN - 1);
}

/*
 * Once the driver has mapped the memory there is no going back: the
 * operation has to be completed, and completing a checkpoint releases the
 * GPU memory, copied or not, while completing a restore resumes the task with
 * whatever was copied in. A task whose copy failed is killed rather than
 * left to run without its GPU memory; CRIU then finds its restore thread
 * gone and fails the operation.
 */
static void lose_task(pid_t pid, const char *what)
{
	pr_err("pid %d's GPU memory could not be %s; killing it rather than running it without\n", pid, what);
	if (kill(pid, SIGKILL) && errno != ESRCH)
		pr_perror("Unable to kill pid %d", pid);
}

/*
 * Runs in the helper. The space the image needs is checked once the sizes
 * are known and before the first byte is copied; a failed copy still
 * completes, with the image left unnamed.
 */
static int checkpoint_helper(pid_t pid, const char *name)
{
	CUcheckpointCustomStorageInfo *info = NULL;
	CUcheckpointCheckpointArgs args = { 0 };
	struct cuda_mem_header hdr = { .magic = CUDA_MEM_MAGIC, .version = CUDA_MEM_VERSION };
	struct cuda_mem_entry *table = NULL;
	char part[80];
	int dir = criu_get_image_dir();
	int fd = -1, ret = -1;
	uint64_t off;
	void *buf = NULL;
	struct statvfs vfs;
	unsigned int i;
	CUresult res;

	if (retain_contexts(&buf))
		return -1;
	args.customStorageInfo_out = &info;
	res = api.checkpoint(pid, &args);
	if (res != CUDA_SUCCESS || !info) {
		pr_err("cuCheckpointProcessCheckpoint to custom storage failed for pid %d: %d\n", pid, res);
		return -1;
	}
	if (info->deviceCount > CUDA_MEM_MAX_GPUS) {
		pr_err("pid %d has GPU memory on %u GPUs\n", pid, info->deviceCount);
		goto complete;
	}
	hdr.count = info->deviceCount;
	table = calloc(hdr.count ? hdr.count : 1, sizeof(*table));
	if (!table)
		goto complete;
	off = align_up(sizeof(hdr) + hdr.count * sizeof(*table));
	for (i = 0; i < hdr.count; i++) {
		table[i].size = info->perDeviceData[i].size;
		table[i].offset = off;
		off = align_up(off + table[i].size);
	}
	if (fstatvfs(dir, &vfs) == 0 && (uint64_t)vfs.f_bavail * vfs.f_frsize < off) {
		pr_err("The images need %" PRIu64 " bytes for pid %d's GPU memory and have %" PRIu64 "\n", off, pid,
		       (uint64_t)vfs.f_bavail * vfs.f_frsize);
		goto complete;
	}
	snprintf(part, sizeof(part), "%s.part", name);
	fd = openat(dir, part, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
	if (fd < 0) {
		pr_perror("Unable to create %s", part);
		goto complete;
	}
	for (i = 0; i < hdr.count; i++) {
		CUcheckpointCustomStoragePerDeviceData *d = &info->perDeviceData[i];
		CUuuid uuid;
		int err;

		if (enter_device(d, &uuid))
			goto complete;
		memcpy(table[i].uuid, uuid.bytes, sizeof(table[i].uuid));
		err = copy_out(fd, d, buf, table[i].offset);
		leave_device();
		if (err)
			goto complete;
		pr_info("Wrote %zu bytes of GPU memory of pid %d\n", d->size, pid);
	}
	if (full_pwrite(fd, &hdr, sizeof(hdr), 0) ||
	    full_pwrite(fd, table, hdr.count * sizeof(*table), sizeof(hdr))) {
		pr_perror("Unable to write %s", part);
		goto complete;
	}
	ret = 0;
complete:
	if (ret)
		lose_task(pid, "saved");
	res = api.complete(info->handle);
	if (res != CUDA_SUCCESS) {
		pr_err("cuCheckpointOperationComplete failed for pid %d: %d\n", pid, res);
		ret = -1;
	}
	if (fd >= 0) {
		if (!ret && (fsync(fd) || renameat(dir, part, dir, name))) {
			pr_perror("Unable to name %s", name);
			ret = -1;
		}
		close(fd);
		if (ret)
			unlinkat(dir, part, 0);
	}
	free(table);
	return ret;
}

/* The image entry for the GPU uuid: the one map pairs it with, or itself. */
static const struct cuda_mem_entry *entry_for(const struct cuda_mem_entry *table, unsigned int count,
					      const CUuuid *uuid, const struct cuda_device_map *map)
{
	const unsigned char *old = uuid->bytes;
	unsigned int i;

	if (map) {
		for (i = 0; i < map->count; i++) {
			if (!memcmp(map->pairs[i].newUuid, uuid->bytes, 16)) {
				old = map->pairs[i].oldUuid;
				break;
			}
		}
	}
	for (i = 0; i < count; i++) {
		if (!memcmp(table[i].uuid, old, 16))
			return &table[i];
	}
	return NULL;
}

static int restore_helper(pid_t pid, const char *name, const struct cuda_device_map *map)
{
	CUcheckpointCustomStorageInfo *info = NULL;
	CUcheckpointRestoreArgs args = { 0 };
	struct cuda_mem_header hdr;
	struct cuda_mem_entry *table = NULL;
	int fd, ret = -1;
	void *buf = NULL;
	unsigned int i;
	CUresult res;

	fd = openat(criu_get_image_dir(), name, O_RDONLY | O_CLOEXEC);
	if (fd < 0) {
		pr_perror("Unable to open %s", name);
		return -1;
	}
	if (full_pread(fd, &hdr, sizeof(hdr), 0) || memcmp(hdr.magic, CUDA_MEM_MAGIC, sizeof(hdr.magic)) ||
	    hdr.version != CUDA_MEM_VERSION || hdr.count > CUDA_MEM_MAX_GPUS) {
		pr_err("%s is not a CUDA memory image this plugin reads\n", name);
		goto out;
	}
	table = calloc(hdr.count ? hdr.count : 1, sizeof(*table));
	if (!table || full_pread(fd, table, hdr.count * sizeof(*table), sizeof(hdr))) {
		pr_err("Unable to read the table of %s\n", name);
		goto out;
	}
	if (retain_contexts(&buf))
		goto out;
	if (map) {
		args.gpuPairs = map->pairs;
		args.gpuPairsCount = map->count;
	}
	args.customStorageInfo_out = &info;
	res = api.restore(pid, &args);
	if (res != CUDA_SUCCESS || !info) {
		pr_err("cuCheckpointProcessRestore from custom storage failed for pid %d: %d\n", pid, res);
		goto out;
	}
	if (info->deviceCount != hdr.count) {
		pr_err("pid %d's GPU memory was on %u GPUs, the restore maps %u\n", pid, hdr.count,
		       info->deviceCount);
		goto complete;
	}
	for (i = 0; i < info->deviceCount; i++) {
		CUcheckpointCustomStoragePerDeviceData *d = &info->perDeviceData[i];
		const struct cuda_mem_entry *e;
		CUuuid uuid;
		int err;

		if (enter_device(d, &uuid))
			goto complete;
		e = entry_for(table, hdr.count, &uuid, map);
		if (!e || e->size != d->size) {
			pr_err("%s has no %zu bytes of GPU memory for this GPU\n", name, d->size);
			leave_device();
			goto complete;
		}
		err = copy_in(fd, d, buf, e->offset);
		leave_device();
		if (err)
			goto complete;
		pr_info("Read %zu bytes of GPU memory of pid %d\n", d->size, pid);
	}
	ret = 0;
complete:
	if (ret)
		lose_task(pid, "restored");
	res = api.complete(info->handle);
	if (res != CUDA_SUCCESS) {
		pr_err("cuCheckpointOperationComplete failed for pid %d: %d\n", pid, res);
		ret = -1;
	}
out:
	free(table);
	close(fd);
	return ret;
}

/*
 * CUDA is initialized and contexts are held in a helper process: in CRIU's,
 * a driver thread would share its fs_struct and break later setns() calls
 * (as for the GPU inventory). CRIU's own SIGCHLD handling does not reap it:
 * a dump has none, and a restore has reset it before RESUME_DEVICES_LATE.
 */
static int run_helper(const char *operation, pid_t pid, int (*fn)(pid_t, const char *, const struct cuda_device_map *),
		      const struct cuda_device_map *map)
{
	struct cuda_wait wait;
	char name[64];
	pid_t child;
	int status, ret;

	if (image_name(pid, name, sizeof(name)))
		return -1;
	child = fork();
	if (child < 0) {
		pr_perror("Unable to fork the CUDA %s helper", operation);
		return -1;
	}
	if (!child)
		_exit(fn(pid, name, map) ? EXIT_FAILURE : EXIT_SUCCESS);

	ret = cuda_wait_init(&wait, operation, pid, 0, NULL, cuda_plugin_timeout);
	if (!ret) {
		/* Not a part of the freeze: the copy takes as long as it takes. */
		wait.ignore_criu_timeout = true;
		ret = cuda_wait_child(&wait, child, 0, &status);
	}
	if (ret) {
		kill(child, SIGKILL);
		waitpid(child, &status, 0);
		return -1;
	}
	if (!WIFEXITED(status) || WEXITSTATUS(status)) {
		pr_err("The CUDA %s helper for pid %d failed (status %#x)\n", operation, pid, status);
		return -1;
	}
	return 0;
}

static int checkpoint_fn(pid_t pid, const char *name, const struct cuda_device_map *map)
{
	return checkpoint_helper(pid, name);
}

static int restore_fn(pid_t pid, const char *name, const struct cuda_device_map *map)
{
	return restore_helper(pid, name, map);
}

int cuda_custom_storage_checkpoint(void *libcuda, pid_t pid)
{
	if (load_api(libcuda)) {
		pr_err("libcuda has no custom storage\n");
		return -1;
	}
	return run_helper("custom-storage-checkpoint", pid, checkpoint_fn, NULL);
}

int cuda_custom_storage_restore(void *libcuda, pid_t pid, const struct cuda_device_map *map)
{
	if (load_api(libcuda)) {
		pr_err("pid %d's GPU memory is in custom storage, which this libcuda does not have\n", pid);
		return -1;
	}
	return run_helper("custom-storage-restore", pid, restore_fn, map);
}
