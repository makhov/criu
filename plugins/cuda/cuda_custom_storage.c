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
#include <pthread.h>
#include <semaphore.h>
#include <stdatomic.h>
#include <stdbool.h>
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
 * GPU's memory in chunks of CUDA_MEM_CHUNK bytes, listed in an index at the
 * GPU's entry's offset. The vpid is the task's pid in its own pid
 * namespace, which a restore gives it again; a checkpoint writes the image
 * under a temporary name and renames it once it is whole, so an image that
 * exists is complete.
 *
 * Where CRIU compresses pages (--compress), the chunks are compressed too,
 * with zstd rather than CRIU's LZ4: GPU memory is mostly a model's weights,
 * bf16 or fp16, which LZ4 does not shrink at all (0.999 of a bf16 model) and
 * an entropy coder does, through their high bytes, sign and exponent. A
 * chunk's high bytes, every second one, are compressed as one zstd frame and
 * its low bytes stored as they are: a bf16 model shrinks to 0.685. A chunk
 * that would not shrink by 1/32 is stored whole. Each chunk is a frame of its
 * own, so a restore decompresses them in parallel.
 *
 * Version 1 images (gcr.3) hold each GPU's memory whole, at its entry's
 * offset, with no index; they are still read.
 */
#define CUDA_MEM_IMAGE	   "cuda-mem-%d.img"
#define CUDA_MEM_MAGIC	   "CRCUDAM1"
#define CUDA_MEM_VERSION   2
#define CUDA_MEM_ALIGN	   4096
#define CUDA_MEM_MAX_GPUS  64
/* The pieces the memory is copied, compressed and indexed in. */
#define CUDA_MEM_CHUNK	   (16UL << 20)
/* Threads a checkpoint or restore copies chunks with, at most. */
#define CUDA_MEM_THREADS   8

struct cuda_mem_header {
	char magic[8];
	uint32_t version;
	uint32_t count;
};

struct cuda_mem_entry {
	unsigned char uuid[16];
	uint64_t size;
	/* Version 2: where the GPU's chunk index is; 1: its memory. */
	uint64_t offset;
};

struct cuda_mem_chunk {
	uint64_t offset;
	/* Bytes of GPU memory: CUDA_MEM_CHUNK, or less in the last chunk. */
	uint32_t size;
	/* 0: stored whole. Else the zstd frame of its high bytes, this many
	 * bytes long, followed by its low bytes.
	 */
	uint32_t packed;
};

struct cuda_storage_api {
	CUresult (*init)(unsigned int flags);
	CUresult (*device_get_count)(int *count);
	CUresult (*device_get)(CUdevice *device, int ordinal);
	CUresult (*device_get_uuid)(CUuuid *uuid, CUdevice device);
	CUresult (*primary_ctx_retain)(CUcontext *ctx, CUdevice device);
	CUresult (*stream_get_ctx)(CUstream stream, CUcontext *ctx);
	CUresult (*stream_create)(CUstream *stream, unsigned int flags);
	CUresult (*stream_destroy)(CUstream stream);
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
		{ "cuStreamCreate", (void **)&api.stream_create },
		{ "cuStreamDestroy_v2", (void **)&api.stream_destroy },
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

/* libzstd, from wherever the node has it: CRIU itself is built without. */
struct zstd_api {
	size_t (*bound)(size_t size);
	void *(*create_cctx)(void);
	size_t (*set_parameter)(void *cctx, int param, int value);
	size_t (*compress2)(void *cctx, void *dst, size_t dst_size, const void *src, size_t src_size);
	size_t (*decompress)(void *dst, size_t dst_size, const void *src, size_t src_size);
	unsigned int (*is_error)(size_t code);
};

#define ZSTD_c_compressionLevel 100
#define ZSTD_c_nbWorkers	400

static int load_zstd(struct zstd_api *z)
{
	void *h = dlopen("libzstd.so.1", RTLD_NOW | RTLD_LOCAL);

	if (!h)
		return -1;
	z->bound = (size_t(*)(size_t))dlsym(h, "ZSTD_compressBound");
	z->create_cctx = (void *(*)(void))dlsym(h, "ZSTD_createCCtx");
	z->set_parameter = (size_t(*)(void *, int, int))dlsym(h, "ZSTD_CCtx_setParameter");
	z->compress2 = (size_t(*)(void *, void *, size_t, const void *, size_t))dlsym(h, "ZSTD_compress2");
	z->decompress = (size_t(*)(void *, size_t, const void *, size_t))dlsym(h, "ZSTD_decompress");
	z->is_error = (unsigned int (*)(size_t))dlsym(h, "ZSTD_isError");
	if (!z->bound || !z->create_cctx || !z->set_parameter || !z->compress2 || !z->decompress || !z->is_error)
		return -1;
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
#define CU_STREAM_NON_BLOCKING	 0x01

/*
 * The driver wants the caller to hold the primary contexts of the GPUs. The
 * n pinned buffers are allocated here too, before the checkpoint or restore
 * starts: a failure now leaves the task as it was.
 */
static int retain_contexts(void **bufs, int n)
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
		pr_err("No CUDA context to allocate pinned buffers in\n");
		return -1;
	}
	for (i = 0; i < n; i++) {
		if (api.mem_host_alloc(&bufs[i], CUDA_MEM_CHUNK, CU_MEMHOSTALLOC_PORTABLE) != CUDA_SUCCESS) {
			pr_err("Unable to allocate a pinned buffer for GPU memory\n");
			leave_device();
			return -1;
		}
	}
	leave_device();
	return 0;
}

/* Makes the context of d's stream current and gives it and the GPU's UUID. */
static int enter_device(const CUcheckpointCustomStoragePerDeviceData *d, CUcontext *ctx, CUuuid *uuid)
{
	CUdevice dev;

	if (api.stream_get_ctx(d->stream, ctx) != CUDA_SUCCESS || api.ctx_push(*ctx) != CUDA_SUCCESS ||
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

/* A chunk's low bytes (lo) and high bytes (hi), and back. */
static void split_bytes(const unsigned char *src, size_t size, unsigned char *hi, unsigned char *lo)
{
	size_t i;

	for (i = 0; i < size / 2; i++) {
		lo[i] = src[2 * i];
		hi[i] = src[2 * i + 1];
	}
}

static void join_bytes(unsigned char *dst, size_t size, const unsigned char *hi, const unsigned char *lo)
{
	size_t i;

	for (i = 0; i < size / 2; i++) {
		dst[2 * i] = lo[i];
		dst[2 * i + 1] = hi[i];
	}
}

static size_t chunks_of(uint64_t size)
{
	return (size + CUDA_MEM_CHUNK - 1) / CUDA_MEM_CHUNK;
}

/*
 * Copies GPU memory out: the main thread copies each chunk off the GPU into
 * a free slot's pinned buffer, and the slot's thread compresses and writes
 * it while the next ones are copied. Records go where the next free bytes
 * are, in whatever order they finish; the index says where.
 */
struct writer;

struct writer_slot {
	struct writer *w;
	void *pinned;
	void *cctx;
	unsigned char *hi, *lo, *out;
	size_t k, len;
	sem_t job, free;
	bool stop, started;
	pthread_t thread;
};

struct writer {
	int fd;
	pthread_mutex_t lock;
	uint64_t pos;
	uint64_t stored;
	/* NULL: chunks are stored whole. */
	struct zstd_api *z;
	struct cuda_mem_chunk *idx;
	atomic_bool failed;
	struct writer_slot slots[CUDA_MEM_THREADS];
	int nslots;
};

static int write_record(struct writer *w, struct cuda_mem_chunk *c, const void *a, size_t alen, const void *b,
			size_t blen)
{
	uint64_t at;
	int err;

	pthread_mutex_lock(&w->lock);
	at = w->pos;
	w->pos += alen + blen;
	w->stored += alen + blen;
	pthread_mutex_unlock(&w->lock);
	c->offset = at;
	if ((err = full_pwrite(w->fd, a, alen, at)) || (blen && (err = full_pwrite(w->fd, b, blen, at + alen)))) {
		pr_err("Unable to write GPU memory: %s\n", strerror(-err));
		return -1;
	}
	return 0;
}

static void store_chunk(struct writer_slot *s)
{
	struct writer *w = s->w;
	struct cuda_mem_chunk *c = &w->idx[s->k];
	size_t len = s->len;

	c->size = len;
	if (w->z && len % 2 == 0) {
		size_t n;

		split_bytes(s->pinned, len, s->hi, s->lo);
		n = w->z->compress2(s->cctx, s->out, w->z->bound(len / 2), s->hi, len / 2);
		if (!w->z->is_error(n) && n < len / 2 - len / 64) {
			c->packed = n;
			if (write_record(w, c, s->out, n, s->lo, len / 2))
				atomic_store(&w->failed, true);
			return;
		}
	}
	c->packed = 0;
	if (write_record(w, c, s->pinned, len, NULL, 0))
		atomic_store(&w->failed, true);
}

static void sem_wait_intr(sem_t *s)
{
	while (sem_wait(s) && errno == EINTR)
		;
}

static void *write_chunks(void *arg)
{
	struct writer_slot *s = arg;

	for (;;) {
		sem_wait_intr(&s->job);
		if (s->stop)
			break;
		store_chunk(s);
		sem_post(&s->free);
	}
	return NULL;
}

/*
 * Sets w up with n slots over the pinned buffers bufs: zstd where CRIU
 * compresses pages and the node has libzstd, and each slot's thread. All of
 * it before the checkpoint starts, so a failure leaves the task as it was.
 */
static int writer_start(struct writer *w, struct zstd_api *z, void **bufs, int n)
{
	int i;

	pthread_mutex_init(&w->lock, NULL);
	atomic_store(&w->failed, false);
	w->nslots = n;
	if (opts.compress_mode) {
		if (load_zstd(z))
			pr_warn("No libzstd.so.1: GPU memory is stored uncompressed\n");
		else
			w->z = z;
	}
	for (i = 0; i < n; i++) {
		struct writer_slot *s = &w->slots[i];

		s->w = w;
		s->pinned = bufs[i];
		sem_init(&s->job, 0, 0);
		sem_init(&s->free, 0, 1);
		if (w->z) {
			s->cctx = w->z->create_cctx();
			s->hi = malloc(CUDA_MEM_CHUNK / 2);
			s->lo = malloc(CUDA_MEM_CHUNK / 2);
			s->out = malloc(w->z->bound(CUDA_MEM_CHUNK / 2));
			if (!s->cctx || !s->hi || !s->lo || !s->out) {
				pr_err("Unable to set up zstd for GPU memory\n");
				return -1;
			}
			w->z->set_parameter(s->cctx, ZSTD_c_compressionLevel, 1);
		}
		if (pthread_create(&s->thread, NULL, write_chunks, s)) {
			pr_err("Unable to start a thread to write GPU memory\n");
			return -1;
		}
		s->started = true;
	}
	return 0;
}

static void writer_stop(struct writer *w)
{
	int i;

	for (i = 0; i < w->nslots; i++) {
		struct writer_slot *s = &w->slots[i];

		if (s->started) {
			s->stop = true;
			sem_post(&s->job);
			pthread_join(s->thread, NULL);
		}
		free(s->hi);
		free(s->lo);
		free(s->out);
	}
}

/* Copies d's memory out and writes its index, whose offset it gives. */
static int copy_out(struct writer *w, const CUcheckpointCustomStoragePerDeviceData *d, uint64_t *index)
{
	size_t n = chunks_of(d->size), k;
	int i, err, ret = -1;

	w->idx = calloc(n ? n : 1, sizeof(*w->idx));
	if (!w->idx)
		return -1;
	for (k = 0; k < n && !atomic_load(&w->failed); k++) {
		struct writer_slot *s = &w->slots[k % w->nslots];
		size_t len = d->size - k * CUDA_MEM_CHUNK < CUDA_MEM_CHUNK ? d->size - k * CUDA_MEM_CHUNK : CUDA_MEM_CHUNK;

		sem_wait_intr(&s->free);
		if (api.memcpy_dtoh(s->pinned, d->devPtr + k * CUDA_MEM_CHUNK, len, d->stream) != CUDA_SUCCESS ||
		    api.stream_synchronize(d->stream) != CUDA_SUCCESS) {
			pr_err("Unable to copy GPU memory out at %zu\n", k * CUDA_MEM_CHUNK);
			atomic_store(&w->failed, true);
			sem_post(&s->free);
			break;
		}
		s->k = k;
		s->len = len;
		sem_post(&s->job);
	}
	/* Every slot done with this GPU's chunks. */
	for (i = 0; i < w->nslots; i++) {
		sem_wait_intr(&w->slots[i].free);
		sem_post(&w->slots[i].free);
	}
	if (atomic_load(&w->failed))
		goto out;
	*index = w->pos;
	if ((err = full_pwrite(w->fd, w->idx, n * sizeof(*w->idx), w->pos))) {
		pr_err("Unable to write GPU memory's index: %s\n", strerror(-err));
		goto out;
	}
	w->pos += n * sizeof(*w->idx);
	ret = 0;
out:
	free(w->idx);
	w->idx = NULL;
	return ret;
}

/* Copies a GPU's chunks back, several at a time. */
struct reader {
	int fd;
	const struct cuda_mem_chunk *idx;
	size_t n;
	CUdeviceptr dev;
	CUcontext ctx;
	const struct zstd_api *z;
	atomic_size_t next;
	atomic_bool failed;
};

struct reader_slot {
	struct reader *r;
	void *pinned;
	unsigned char *packed, *hi;
	pthread_t thread;
};

static void *read_chunks(void *arg)
{
	struct reader_slot *s = arg;
	struct reader *r = s->r;
	CUstream stream = NULL;
	bool entered = false;
	size_t k;

	if (api.ctx_push(r->ctx) != CUDA_SUCCESS || !(entered = true) ||
	    api.stream_create(&stream, CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS) {
		pr_err("Unable to set up a CUDA stream to restore GPU memory\n");
		atomic_store(&r->failed, true);
		goto out;
	}
	while (!atomic_load(&r->failed) && (k = atomic_fetch_add(&r->next, 1)) < r->n) {
		const struct cuda_mem_chunk *c = &r->idx[k];
		size_t half = c->size / 2;

		if (!c->packed) {
			if (full_pread(r->fd, s->pinned, c->size, c->offset))
				goto fail;
		} else {
			if (!r->z || full_pread(r->fd, s->packed, c->packed + half, c->offset) ||
			    r->z->decompress(s->hi, half, s->packed, c->packed) != half)
				goto fail;
			join_bytes(s->pinned, c->size, s->hi, s->packed + c->packed);
		}
		if (api.memcpy_htod(r->dev + k * CUDA_MEM_CHUNK, s->pinned, c->size, stream) != CUDA_SUCCESS ||
		    api.stream_synchronize(stream) != CUDA_SUCCESS)
			goto fail;
		continue;
fail:
		pr_err("Unable to restore chunk %zu of GPU memory\n", k);
		atomic_store(&r->failed, true);
	}
out:
	if (stream)
		api.stream_destroy(stream);
	if (entered)
		leave_device();
	return NULL;
}

static int copy_in(struct reader *r, struct reader_slot *slots, int nslots)
{
	int i, started = 0;

	atomic_store(&r->next, 0);
	atomic_store(&r->failed, false);
	for (i = 0; i < nslots; i++) {
		slots[i].r = r;
		if (pthread_create(&slots[i].thread, NULL, read_chunks, &slots[i])) {
			pr_err("Unable to start a thread to restore GPU memory\n");
			atomic_store(&r->failed, true);
			break;
		}
		started++;
	}
	for (i = 0; i < started; i++)
		pthread_join(slots[i].thread, NULL);
	return atomic_load(&r->failed) || !started ? -1 : 0;
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
	struct writer w = { .fd = -1 };
	void *bufs[CUDA_MEM_THREADS] = {};
	struct zstd_api z;
	char part[80];
	int dir = criu_get_image_dir();
	long cpus = sysconf(_SC_NPROCESSORS_ONLN);
	int nslots = cpus > CUDA_MEM_THREADS ? CUDA_MEM_THREADS : cpus > 1 ? (int)cpus : 1;
	int ret = -1;
	uint64_t need, raw = 0;
	struct statvfs vfs;
	unsigned int i;
	CUresult res;

	if (retain_contexts(bufs, nslots) || writer_start(&w, &z, bufs, nslots))
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
	/* What it takes uncompressed, which compression only lessens. */
	w.pos = need = align_up(sizeof(hdr) + hdr.count * sizeof(*table));
	for (i = 0; i < hdr.count; i++)
		need += info->perDeviceData[i].size + chunks_of(info->perDeviceData[i].size) * sizeof(struct cuda_mem_chunk);
	if (fstatvfs(dir, &vfs) == 0 && (uint64_t)vfs.f_bavail * vfs.f_frsize < need) {
		pr_err("The images need %" PRIu64 " bytes for pid %d's GPU memory and have %" PRIu64 "\n", need, pid,
		       (uint64_t)vfs.f_bavail * vfs.f_frsize);
		goto complete;
	}
	snprintf(part, sizeof(part), "%s.part", name);
	w.fd = openat(dir, part, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
	if (w.fd < 0) {
		pr_perror("Unable to create %s", part);
		goto complete;
	}
	for (i = 0; i < hdr.count; i++) {
		CUcheckpointCustomStoragePerDeviceData *d = &info->perDeviceData[i];
		CUcontext ctx;
		CUuuid uuid;
		int err;

		if (enter_device(d, &ctx, &uuid))
			goto complete;
		memcpy(table[i].uuid, uuid.bytes, sizeof(table[i].uuid));
		table[i].size = d->size;
		err = copy_out(&w, d, &table[i].offset);
		leave_device();
		if (err)
			goto complete;
		raw += d->size;
	}
	if (full_pwrite(w.fd, &hdr, sizeof(hdr), 0) ||
	    full_pwrite(w.fd, table, hdr.count * sizeof(*table), sizeof(hdr))) {
		pr_perror("Unable to write %s", part);
		goto complete;
	}
	pr_info("Wrote %" PRIu64 " bytes of GPU memory of pid %d as %" PRIu64 "%s\n", raw, pid, w.stored,
		w.z ? " (zstd)" : "");
	ret = 0;
complete:
	writer_stop(&w);
	if (ret)
		lose_task(pid, "saved");
	res = api.complete(info->handle);
	if (res != CUDA_SUCCESS) {
		pr_err("cuCheckpointOperationComplete failed for pid %d: %d\n", pid, res);
		ret = -1;
	}
	if (w.fd >= 0) {
		if (!ret && (fsync(w.fd) || renameat(dir, part, dir, name))) {
			pr_perror("Unable to name %s", name);
			ret = -1;
		}
		close(w.fd);
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

/*
 * The chunks of e's memory: its index (version 2), or, in a version 1 image,
 * its memory whole, cut in the same chunks. Checked against e's size, so a
 * restore that starts can copy all of it.
 */
static struct cuda_mem_chunk *read_index(int fd, uint32_t version, const struct cuda_mem_entry *e, bool *packed)
{
	size_t n = chunks_of(e->size), k;
	struct cuda_mem_chunk *idx = calloc(n ? n : 1, sizeof(*idx));

	if (!idx)
		return NULL;
	if (version == 1) {
		for (k = 0; k < n; k++) {
			idx[k].offset = e->offset + k * CUDA_MEM_CHUNK;
			idx[k].size = e->size - k * CUDA_MEM_CHUNK < CUDA_MEM_CHUNK ? e->size - k * CUDA_MEM_CHUNK :
										     CUDA_MEM_CHUNK;
		}
		return idx;
	}
	if (full_pread(fd, idx, n * sizeof(*idx), e->offset))
		goto bad;
	for (k = 0; k < n; k++) {
		uint64_t want = e->size - k * CUDA_MEM_CHUNK < CUDA_MEM_CHUNK ? e->size - k * CUDA_MEM_CHUNK : CUDA_MEM_CHUNK;

		if (idx[k].size != want || (idx[k].packed && (idx[k].size % 2 || idx[k].packed >= idx[k].size / 2)))
			goto bad;
		*packed |= idx[k].packed != 0;
	}
	return idx;
bad:
	free(idx);
	return NULL;
}

static int restore_helper(pid_t pid, const char *name, const struct cuda_device_map *map)
{
	CUcheckpointCustomStorageInfo *info = NULL;
	CUcheckpointRestoreArgs args = { 0 };
	struct reader_slot slots[CUDA_MEM_THREADS] = {};
	void *bufs[CUDA_MEM_THREADS] = {};
	struct cuda_mem_chunk **indexes = NULL;
	struct cuda_mem_header hdr;
	struct cuda_mem_entry *table = NULL;
	struct zstd_api z, *zp = NULL;
	long cpus = sysconf(_SC_NPROCESSORS_ONLN);
	int nslots = cpus > CUDA_MEM_THREADS ? CUDA_MEM_THREADS : cpus > 1 ? (int)cpus : 1;
	bool packed = false;
	int fd, ret = -1;
	unsigned int i;
	CUresult res;

	fd = openat(criu_get_image_dir(), name, O_RDONLY | O_CLOEXEC);
	if (fd < 0) {
		pr_perror("Unable to open %s", name);
		return -1;
	}
	if (full_pread(fd, &hdr, sizeof(hdr), 0) || memcmp(hdr.magic, CUDA_MEM_MAGIC, sizeof(hdr.magic)) ||
	    (hdr.version != 1 && hdr.version != CUDA_MEM_VERSION) || hdr.count > CUDA_MEM_MAX_GPUS) {
		pr_err("%s is not a CUDA memory image this plugin reads\n", name);
		goto out;
	}
	table = calloc(hdr.count ? hdr.count : 1, sizeof(*table));
	indexes = calloc(hdr.count ? hdr.count : 1, sizeof(*indexes));
	if (!table || !indexes || full_pread(fd, table, hdr.count * sizeof(*table), sizeof(hdr))) {
		pr_err("Unable to read the table of %s\n", name);
		goto out;
	}
	/* Everything the copy needs is read and set up before the restore
	 * starts, after which a failure costs the task its GPU memory.
	 */
	for (i = 0; i < hdr.count; i++) {
		indexes[i] = read_index(fd, hdr.version, &table[i], &packed);
		if (!indexes[i]) {
			pr_err("The index of GPU %u in %s is damaged\n", i, name);
			goto out;
		}
	}
	if (packed) {
		if (load_zstd(&z)) {
			pr_err("%s is compressed with zstd, and there is no libzstd.so.1\n", name);
			goto out;
		}
		zp = &z;
	}
	if (retain_contexts(bufs, nslots))
		goto out;
	for (i = 0; i < (unsigned int)nslots; i++) {
		slots[i].pinned = bufs[i];
		if (packed) {
			slots[i].packed = malloc(CUDA_MEM_CHUNK);
			slots[i].hi = malloc(CUDA_MEM_CHUNK / 2);
			if (!slots[i].packed || !slots[i].hi)
				goto out;
		}
	}
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
		struct reader r = { .fd = fd, .z = zp, .dev = d->devPtr };
		CUuuid uuid;
		int err;

		if (enter_device(d, &r.ctx, &uuid))
			goto complete;
		leave_device();
		e = entry_for(table, hdr.count, &uuid, map);
		if (!e || e->size != d->size) {
			pr_err("%s has no %zu bytes of GPU memory for this GPU\n", name, d->size);
			goto complete;
		}
		r.idx = indexes[e - table];
		r.n = chunks_of(e->size);
		err = copy_in(&r, slots, nslots);
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
	for (i = 0; i < CUDA_MEM_THREADS; i++) {
		free(slots[i].packed);
		free(slots[i].hi);
	}
	if (indexes)
		for (i = 0; i < hdr.count; i++)
			free(indexes[i]);
	free(indexes);
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
