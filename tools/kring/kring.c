// kring: CUPTI injection library that answers "which kernel is this GPU parked in?" during a hang
// WITHOUT attaching a debugger.
//
// Load at process start:  CUDA_INJECTION64_PATH=/opt/kring/libkring.so  (KRING_DIR default /tmp/kring)
// Per CUDA process it mmaps KRING_DIR/kring-<pid>.bin holding two rings:
//   launches    : every kernel / graph launch API call (name, stream, correlation id, host ns)
//   completions : every finished kernel (CUPTI activity CONCURRENT_KERNEL, flushed every KRING_FLUSH_MS)
// A launch whose correlation id never shows up in completions (and is older than the last completion
// on its stream) is what the GPU is stuck on. Read with spark-kring (python, no attach needed).
// KRING_ACTIVITY=0 keeps only the launch ring (cheaper, but no completion info).
//
// KRING_MODE=profile (profiling boots only) adds a kernel PROFILER. With KRING_ACTIVITY=0 the launch-only hang ring
// stays on too; otherwise the profiler runs alone (no API callbacks). CUPTI activity (kernels, memcpy, memset) is enabled ONLY while KRING_DIR/ctl holds a label other than
// "off" (poll every 50 ms). Every record goes to KRING_DIR/kprof-<pid>.bin (40 B each); names and the
// armed windows (label + CUPTI timestamps) go to kprof-<pid>.txt. KRING_MAX_MB (default 1024) caps the .bin.
// Report: spark-kprof.
#define _GNU_SOURCE
#include <cupti.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define NAME_LEN 160
#define N_LAUNCH 16384
#define N_DONE 16384
#define MAGIC 0x4b52494e47303031ULL  // "KRING001"

typedef struct {
  uint64_t seq, host_ns;
  uint64_t stream;       // CUstream / cudaStream_t handle as seen by the API
  uint32_t corr, cbid;   // correlation id, CUPTI callback id
  uint32_t domain, pad;  // 1 = runtime, 2 = driver
  char name[NAME_LEN];
} launch_t;

typedef struct {
  uint64_t seq, start, end;  // GPU timestamps (ns)
  uint32_t corr, stream_id;
  uint32_t graph_id, pad;
  char name[NAME_LEN];
} done_t;

typedef struct {
  uint64_t magic, pid, n_launch, n_done, launch_size, done_size;
  volatile uint64_t launch_head, done_head, dropped_records, flushes;
  uint64_t reserved[6];
  launch_t launches[N_LAUNCH];
  done_t done[N_DONE];
} ring_t;

static ring_t *R;
static __thread int in_runtime_launch;  // runtime launches call the driver: log them once
static uint32_t flush_ms = 100;
static CUpti_SubscriberHandle sub;

static uint64_t now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

static void copy_name(char *dst, const char *src) {
  if (!src) { dst[0] = 0; return; }
  strncpy(dst, src, NAME_LEN - 1);
  dst[NAME_LEN - 1] = 0;
}

static uint64_t stream_of(CUpti_CallbackDomain d, CUpti_CallbackId id, const void *p) {
  if (!p) return 0;
  if (d == CUPTI_CB_DOMAIN_DRIVER_API) {
    switch (id) {
      case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel:
      case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz:
        return (uint64_t)((const cuLaunchKernel_params *)p)->hStream;
      case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx:
      case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz: {
        const CUlaunchConfig *c = ((const cuLaunchKernelEx_params *)p)->config;
        return c ? (uint64_t)c->hStream : 0;
      }
      case CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch:
      case CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch_ptsz:
        return (uint64_t)((const cuGraphLaunch_params *)p)->hStream;
    }
  } else {
    switch (id) {
      case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000:
        return (uint64_t)((const cudaLaunchKernel_v7000_params *)p)->stream;
      case CUPTI_RUNTIME_TRACE_CBID_cudaGraphLaunch_v10000:
        return (uint64_t)((const cudaGraphLaunch_v10000_params *)p)->stream;
    }
  }
  return 0;
}

static void CUPTIAPI on_api(void *ud, CUpti_CallbackDomain d, CUpti_CallbackId id, const void *data) {
  const CUpti_CallbackData *cb = (const CUpti_CallbackData *)data;
  if (!R) return;
  if (d == CUPTI_CB_DOMAIN_RUNTIME_API) {
    if (cb->callbackSite == CUPTI_API_EXIT) { in_runtime_launch = 0; return; }
    in_runtime_launch = 1;
  } else if (cb->callbackSite != CUPTI_API_ENTER || in_runtime_launch) {
    return;
  }
  uint64_t s = __atomic_fetch_add(&R->launch_head, 1, __ATOMIC_RELAXED);
  launch_t *e = &R->launches[s % N_LAUNCH];
  e->seq = 0;  // mark in-progress for readers
  e->host_ns = now_ns();
  e->stream = stream_of(d, id, cb->functionParams);
  e->corr = cb->correlationId;
  e->cbid = id;
  e->domain = d == CUPTI_CB_DOMAIN_RUNTIME_API ? 1 : 2;
  copy_name(e->name, cb->symbolName ? cb->symbolName : cb->functionName);
  __atomic_store_n(&e->seq, s + 1, __ATOMIC_RELEASE);
}

static void CUPTIAPI buf_req(uint8_t **buf, size_t *size, size_t *maxrec) {
  *size = 8 << 20;
  *buf = (uint8_t *)aligned_alloc(8, *size);
  *maxrec = 0;
}

static void CUPTIAPI buf_done(CUcontext ctx, uint32_t stream, uint8_t *buf, size_t size, size_t valid) {
  CUpti_Activity *rec = NULL;
  while (R && cuptiActivityGetNextRecord(buf, valid, &rec) == CUPTI_SUCCESS) {
    if (rec->kind != CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL && rec->kind != CUPTI_ACTIVITY_KIND_KERNEL) continue;
    const CUpti_ActivityKernel10 *k = (const CUpti_ActivityKernel10 *)rec;
    uint64_t s = __atomic_fetch_add(&R->done_head, 1, __ATOMIC_RELAXED);
    done_t *e = &R->done[s % N_DONE];
    e->seq = 0;
    e->start = k->start;
    e->end = k->end;
    e->corr = k->correlationId;
    e->stream_id = k->streamId;
    e->graph_id = k->graphId;
    copy_name(e->name, k->name);
    __atomic_store_n(&e->seq, s + 1, __ATOMIC_RELEASE);
  }
  size_t dropped = 0;
  if (cuptiActivityGetNumDroppedRecords(ctx, stream, &dropped) == CUPTI_SUCCESS && R)
    __atomic_fetch_add(&R->dropped_records, dropped, __ATOMIC_RELAXED);
  if (R) __atomic_fetch_add(&R->flushes, 1, __ATOMIC_RELAXED);
  free(buf);
}

// CUPTI's own periodic flush only hands over FULL buffers; force partial ones out so completions
// reach the ring within ~flush_ms even when the GPU is stuck.
static void *flusher(void *arg) {
  (void)arg;
  for (;;) {
    usleep(flush_ms * 1000);
    cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
  }
  return NULL;
}

// ---------------------------------------------------------------- profile mode
typedef struct {
  uint64_t start, end;
  uint32_t name_id, corr, stream_id, graph_id;
  uint32_t kind, device;  // kind 0 kernel, 1 memcpy, 2 memset
} prec_t;

#define NHASH (1 << 16)
static const char *htab_key[NHASH];
static uint32_t htab_id[NHASH], n_names;
static pthread_mutex_t pmu = PTHREAD_MUTEX_INITIALIZER;
static FILE *pbin, *ptxt;
static uint64_t pbytes, pmax;
static int parmed, pcapped;

static uint32_t name_id(const char *n) {  // caller holds pmu
  if (!n) n = "?";
  uint32_t h = 2166136261u;
  for (const char *c = n; *c; c++) h = (h ^ (uint8_t)*c) * 16777619u;
  for (uint32_t i = h & (NHASH - 1);; i = (i + 1) & (NHASH - 1)) {
    if (!htab_key[i]) {
      if (n_names >= NHASH - 1) return 0;
      htab_key[i] = strdup(n);
      htab_id[i] = ++n_names;
      fprintf(ptxt, "N\t%u\t%s\n", n_names, n);
      return n_names;
    }
    if (strcmp(htab_key[i], n) == 0) return htab_id[i];
  }
}

static const char *CPY[] = {"?", "HtoD", "DtoH", "HtoA", "AtoH", "AtoA", "AtoD", "DtoA", "DtoD", "HtoH", "PtoP"};

static void CUPTIAPI pbuf_done(CUcontext ctx, uint32_t stream, uint8_t *buf, size_t size, size_t valid) {
  CUpti_Activity *rec = NULL;
  pthread_mutex_lock(&pmu);
  while (cuptiActivityGetNextRecord(buf, valid, &rec) == CUPTI_SUCCESS) {
    prec_t r = {0};
    char nb[48];
    if (rec->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL || rec->kind == CUPTI_ACTIVITY_KIND_KERNEL) {
      const CUpti_ActivityKernel10 *k = (const CUpti_ActivityKernel10 *)rec;
      r = (prec_t){k->start, k->end, name_id(k->name), k->correlationId, k->streamId, k->graphId, 0, k->deviceId};
    } else if (rec->kind == CUPTI_ACTIVITY_KIND_MEMCPY) {
      const CUpti_ActivityMemcpy6 *m = (const CUpti_ActivityMemcpy6 *)rec;
      snprintf(nb, sizeof nb, "[memcpy %s]", CPY[m->copyKind < 11 ? m->copyKind : 0]);
      r = (prec_t){m->start, m->end, name_id(nb), m->correlationId, m->streamId, m->graphId, 1, m->deviceId};
    } else if (rec->kind == CUPTI_ACTIVITY_KIND_MEMSET) {
      const CUpti_ActivityMemset4 *m = (const CUpti_ActivityMemset4 *)rec;
      r = (prec_t){m->start, m->end, name_id("[memset]"), m->correlationId, m->streamId, m->graphId, 2, m->deviceId};
    } else {
      continue;
    }
    if (pbytes + sizeof r > pmax) {
      if (!pcapped) fprintf(ptxt, "X\tcap %llu MB reached\n", (unsigned long long)(pmax >> 20));
      pcapped = 1;
      continue;
    }
    fwrite(&r, sizeof r, 1, pbin);
    pbytes += sizeof r;
  }
  size_t dropped = 0;
  if (cuptiActivityGetNumDroppedRecords(ctx, stream, &dropped) == CUPTI_SUCCESS && dropped)
    fprintf(ptxt, "D\t%zu\n", dropped);
  pthread_mutex_unlock(&pmu);
  free(buf);
}

static const CUpti_ActivityKind PKINDS[] = {CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL, CUPTI_ACTIVITY_KIND_MEMCPY,
                                            CUPTI_ACTIVITY_KIND_MEMSET};

// Polls KRING_DIR/ctl. Label != "off" arms activity; each label change writes a window line
// "P <label> <cupti_ts> <realtime_ns>"; "off" flushes, disarms and writes "S <cupti_ts> <realtime_ns>".
static void *pctl(void *arg) {
  const char *ctl = (const char *)arg;
  char cur[64] = "off";
  for (;;) {
    usleep(parmed ? 20000 : 50000);
    char lab[64] = "off";
    FILE *f = fopen(ctl, "r");
    if (f) {
      if (fscanf(f, "%63s", lab) != 1) strcpy(lab, "off");
      fclose(f);
    }
    if (strcmp(lab, cur) == 0) {
      if (parmed) cuptiActivityFlushAll(0);  // hand over full buffers only; cheap
      continue;
    }
    uint64_t ts = 0;
    cuptiGetTimestamp(&ts);
    if (strcmp(lab, "off") == 0) {
      for (unsigned i = 0; i < sizeof PKINDS / sizeof *PKINDS; i++) cuptiActivityDisable(PKINDS[i]);
      cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
      pthread_mutex_lock(&pmu);
      fprintf(ptxt, "S\t%llu\t%llu\n", (unsigned long long)ts, (unsigned long long)now_ns());
      fflush(pbin); fflush(ptxt);
      pthread_mutex_unlock(&pmu);
      parmed = 0;
    } else {
      pthread_mutex_lock(&pmu);
      fprintf(ptxt, "P\t%s\t%llu\t%llu\n", lab, (unsigned long long)ts, (unsigned long long)now_ns());
      fflush(ptxt);
      pthread_mutex_unlock(&pmu);
      if (!parmed) {
        static int registered;
        if (!registered) {  // deferred to the first arm so an unarmed process carries no CUPTI activity state
          registered = 1;
          cuptiActivityRegisterCallbacks(buf_req, pbuf_done);
        }
        for (unsigned i = 0; i < sizeof PKINDS / sizeof *PKINDS; i++) cuptiActivityEnable(PKINDS[i]);
        // KRING_HWTRACE=1: try Blackwell HES kernel timestamps (cheaper than SW instrumentation). Opt-in, untested:
        // on GB10 a call before the first activity enable returned 15 (NOT_INITIALIZED), 2026-09-27.
        const char *hw = getenv("KRING_HWTRACE");
        if (hw && strcmp(hw, "1") == 0) {
          int rc = (int)cuptiActivityEnableHWTrace(1);
          pthread_mutex_lock(&pmu);
          fprintf(ptxt, "W\thwtrace %s (rc %d)\n", rc == 0 ? "on" : "off", rc);
          pthread_mutex_unlock(&pmu);
        }
      }
      parmed = 1;
    }
    strcpy(cur, lab);
  }
  return NULL;
}

static int profile_init(const char *dir) {
  char path[512];
  snprintf(path, sizeof path, "%s/kprof-%d.bin", dir, (int)getpid());
  pbin = fopen(path, "wb");
  snprintf(path, sizeof path, "%s/kprof-%d.txt", dir, (int)getpid());
  ptxt = fopen(path, "w");
  if (!pbin || !ptxt) { perror("kring profile open"); return 1; }  // never 0: see the return note below
  setvbuf(pbin, NULL, _IOFBF, 1 << 20);
  const char *mb = getenv("KRING_MAX_MB");
  pmax = (uint64_t)(mb && atoi(mb) > 0 ? atoi(mb) : 1024) << 20;
  char exe[256] = "";
  ssize_t n = readlink("/proc/self/exe", exe, sizeof exe - 1);
  if (n > 0) exe[n] = 0;
  fprintf(ptxt, "H\tkprof1\tpid %d\trec %zu\texe %s\n", (int)getpid(), sizeof(prec_t), exe);
  fflush(ptxt);
  static char ctl[512];
  snprintf(ctl, sizeof ctl, "%s/ctl", dir);
  pthread_t th;
  pthread_create(&th, NULL, pctl, ctl);
  pthread_detach(th);
  fprintf(stderr, "kring: PROFILE mode, idle until %s holds a label (records -> %s/kprof-%d.*)\n", ctl, dir, (int)getpid());
  return 1;
}

int InitializeInjection(void) {
  const char *dir = getenv("KRING_DIR");
  if (!dir || !*dir) dir = "/tmp/kring";
  mkdir(dir, 0755);
  const char *mode = getenv("KRING_MODE");
  const char *act = getenv("KRING_ACTIVITY");
  int prof = mode && strcmp(mode, "profile") == 0;
  if (prof) {
    profile_init(dir);
    // profile + KRING_ACTIVITY=0: keep the launch-only hang ring as well (its completions would need the
    // activity API, which the profiler owns). Any other KRING_ACTIVITY: profiler only.
    if (!act || strcmp(act, "0") != 0) return 1;
  }
  char path[512];
  snprintf(path, sizeof path, "%s/kring-%d.bin", dir, (int)getpid());
  int fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0644);
  if (fd < 0) { perror("kring open"); return prof; }
  if (ftruncate(fd, sizeof(ring_t)) != 0) { perror("kring ftruncate"); close(fd); return prof; }
  R = (ring_t *)mmap(NULL, sizeof(ring_t), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  if (R == MAP_FAILED) { R = NULL; perror("kring mmap"); return prof; }
  R->pid = getpid(); R->n_launch = N_LAUNCH; R->n_done = N_DONE;
  R->launch_size = sizeof(launch_t); R->done_size = sizeof(done_t);
  __atomic_store_n(&R->magic, MAGIC, __ATOMIC_RELEASE);

  if (cuptiSubscribe(&sub, (CUpti_CallbackFunc)on_api, NULL) != CUPTI_SUCCESS) {
    fprintf(stderr, "kring: cuptiSubscribe failed (another CUPTI subscriber?)\n");
    return prof;
  }
  static const CUpti_CallbackId drv[] = {
      CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz,
      CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz,
      CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel, CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch,
      CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch_ptsz};
  static const CUpti_CallbackId rt[] = {
      CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000, CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_v11060,
      CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernel_v9000, CUPTI_RUNTIME_TRACE_CBID_cudaGraphLaunch_v10000};
  for (unsigned i = 0; i < sizeof drv / sizeof *drv; i++) cuptiEnableCallback(1, sub, CUPTI_CB_DOMAIN_DRIVER_API, drv[i]);
  for (unsigned i = 0; i < sizeof rt / sizeof *rt; i++) cuptiEnableCallback(1, sub, CUPTI_CB_DOMAIN_RUNTIME_API, rt[i]);

  if (!prof && (!act || strcmp(act, "0") != 0)) {
    cuptiActivityRegisterCallbacks(buf_req, buf_done);
    if (cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) != CUPTI_SUCCESS)
      fprintf(stderr, "kring: activity enable failed; launch ring only\n");
    const char *fm = getenv("KRING_FLUSH_MS");
    if (fm && atoi(fm) > 0) flush_ms = (uint32_t)atoi(fm);
    pthread_t th;
    pthread_create(&th, NULL, flusher, NULL);
    pthread_detach(th);
  }
  fprintf(stderr, "kring: tracing kernel launches into %s (activity=%s)\n", path, act ? act : "1");
  return 1;  // libcuda: non-zero = success; 0 makes it dlclose us (and CUPTI then jumps into unmapped code)
}
