# Dependencies

valubench is built to need almost nothing. The binary links only against the C
library:

```
$ ldd build/valubench
    linux-vdso.so.1
    libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6
    /lib64/ld-linux-x86-64.so.2
```

Everything optional — OpenCL, NVML — is `dlopen`'d at runtime. A machine with no
GPU runs the same binary and reports CPU results; nothing needs to be recompiled
and nothing fails to start.

**Package names differ between distributions and drift between releases.** What
does not drift is which *files* have to exist, so each section below states the
requirement first and the packages second. When a package name here is wrong for
your distro, check for the file.

---

## Minimum: build and run on CPU

**Needs:** a C11 compiler, `make`, and libc headers. Nothing else to build and
run. `make check` also needs `python3` (stdlib only): three of its checks read
the binary's JSON with it, and fail saying so when it is missing.

```bash
# Debian / Ubuntu
sudo apt install build-essential

# RHEL / Rocky / Alma / Fedora / Amazon Linux
sudo dnf install gcc make glibc-devel
```

Clang works equally well and is fully tested — `make CC=clang`. Install
`clang` (both families use that name).

```bash
make && ./build/valubench
```

That is the whole requirement for CPU benchmarking. Everything below adds
capability, and its absence is reported rather than fatal.

---

## Optional: authoritative OpenCL headers

**Needs:** `/usr/include/CL/cl.h`.

Without it the build uses the declarations in `include/vb_cl.h` and behaves
identically. With it the build uses the real Khronos types. `make config`
reports which was chosen.

```bash
# Debian / Ubuntu
sudo apt install opencl-headers

# RHEL / Fedora
sudo dnf install opencl-headers
```

You do **not** need `ocl-icd-opencl-dev` (Debian) or `ocl-icd-devel` (RHEL).
Those provide the `libOpenCL.so` link symlink, which matters only if you link
against OpenCL — valubench deliberately does not.

---

## GPU support

Two files must exist at **run** time:

| File | Provided by |
|---|---|
| `libOpenCL.so.1` | the ICD loader |
| at least one `/etc/OpenCL/vendors/*.icd` | your GPU vendor's runtime |

### The loader

```bash
# Debian / Ubuntu
sudo apt install ocl-icd-libopencl1

# RHEL / Fedora
sudo dnf install ocl-icd
```

### The vendor runtime

**Intel** (integrated graphics, Arc, Data Center GPU)

```bash
sudo apt install intel-opencl-icd            # Debian / Ubuntu
sudo dnf install intel-compute-runtime       # Fedora
```

**NVIDIA** — the driver already contains it. `libnvidia-opencl.so.1` and
`/etc/OpenCL/vendors/nvidia.icd` ship with the driver package, so if `nvidia-smi`
works you almost certainly have OpenCL. On a bare distro image:

```bash
sudo apt install nvidia-driver-550           # or the version for your card
sudo dnf install nvidia-driver
```

**AMD** — either the ROCm stack, or Mesa's OpenCL for older or lighter setups:

```bash
sudo apt install mesa-opencl-icd             # Mesa (rusticl/clover)
# or ROCm, from AMD's repository:
sudo apt install rocm-opencl-runtime
sudo dnf install rocm-opencl                 # RHEL / Fedora
```

### Permissions

Rendering nodes are group-restricted. An unprivileged user needs membership,
or OpenCL will report **zero platforms** even with the driver installed:

```bash
sudo usermod -aG render,video $USER          # then log out and back in
```

For an immediate grant without re-login (`/dev` is devtmpfs, so ACLs work here):

```bash
sudo setfacl -m u:$USER:rw /dev/dri/renderD128
```

Verify:

```bash
./build/valubench --list-devices
```

It explains the reason when it finds nothing, rather than just reporting none.

---

## Energy measurement

Optional. Absent counters are reported with a reason, never fatal.

**CPU (Intel and recent AMD): RAPL.** Needs the kernel `powercap` interface,
which is standard, plus read permission. RAPL is root-only on most modern
distributions as hardening against the PLATYPUS side channel:

```bash
sudo chmod a+r /sys/class/powercap/intel-rapl:*/energy_uj
```

`setfacl` cannot be used here — sysfs has no ACL support. The mode also resets
on reboot, since sysfs is rebuilt each boot; for persistence use a udev rule
(`MODE=` applies only to device nodes, not sysfs attributes):

```
# /etc/udev/rules.d/99-rapl-readable.rules
SUBSYSTEM=="powercap", ACTION=="add", \
  RUN+="/bin/sh -c 'chmod a+r /sys%p/energy_uj 2>/dev/null || true'"
```

Running the benchmark as root also works and changes no system state.

On client Intel parts the RAPL `uncore` domain **is** the integrated GPU, so one
run yields both a CPU and a device reading.

**NVIDIA GPU: NVML.** `libnvidia-ml.so.1`, part of the driver. No extra package.

**AMD and discrete Intel GPU: DRM hwmon.** `energy1_input` or `power1_average`
under the card's hwmon node, provided by the `amdgpu` or `i915`/`xe` kernel
driver. No extra package.

---

## Optional tooling

```bash
sudo apt install clinfo python3              # Debian / Ubuntu
sudo dnf install clinfo python3              # RHEL / Fedora
```

`clinfo` is useful for diagnosing an OpenCL install independently of valubench.
`python3` (stdlib only, no pip packages) is needed for
[`tools/sweep.py`](../tools/sweep.py) and the other scripts in `tools/`, and by
`make check`; the benchmark itself does not use it.

---

## Copy-paste for a fresh cloud instance

**Ubuntu / Debian, NVIDIA instance** (AWS `g4dn`, `g5`, `g6`, `p3`, `p4`):

```bash
sudo apt update
sudo apt install -y build-essential opencl-headers ocl-icd-libopencl1 clinfo
# driver, if the image does not already have one (check with: nvidia-smi)
sudo apt install -y nvidia-driver-550
sudo chmod a+r /sys/class/powercap/intel-rapl:*/energy_uj 2>/dev/null || true
make && ./build/valubench --list-devices && ./build/valubench
```

**Ubuntu / Debian, AMD instance** (AWS `g4ad`):

```bash
sudo apt update
sudo apt install -y build-essential opencl-headers ocl-icd-libopencl1 \
                    mesa-opencl-icd clinfo
sudo usermod -aG render,video $USER          # re-login, or use setfacl above
make && ./build/valubench --list-devices && ./build/valubench
```

**RHEL / Rocky / Amazon Linux 2023, NVIDIA:**

```bash
sudo dnf install -y gcc make glibc-devel opencl-headers ocl-icd clinfo
# NVIDIA driver from the vendor repo, or use a GPU-enabled AMI
make && ./build/valubench --list-devices && ./build/valubench
```

Amazon Linux 2023 carries fewer OpenCL packages than Fedora; `ocl-icd` may need
EPEL, and the NVIDIA driver is usually simpler to get from a GPU-enabled AMI
than to install by hand.

**CPU only, anywhere:**

```bash
sudo apt install -y build-essential     # or: dnf install -y gcc make glibc-devel
make && ./build/valubench
```

---

## Nothing here is needed at run time on the target

Every package above is install-time. The benchmark itself opens no network
connection, downloads nothing, and reads no configuration file — which is the
point, given it is meant to run on freshly delivered hardware in a bare
environment.

## Checking what you got

```bash
make config              # compiler, which ISA paths built, CL headers or not
./build/valubench --list          # kernels, and which are available here
./build/valubench --list-devices  # OpenCL devices, or why there are none
make check                        # known-answer vectors + every kernel
```
