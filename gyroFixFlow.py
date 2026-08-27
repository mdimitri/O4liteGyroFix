import os
import struct
import numpy as np
# import matplotlib.pyplot as plt
from tkinter import filedialog
from tqdm import tqdm
from scipy.spatial.transform import Rotation
from scipy.signal import savgol_filter, medfilt
from scipy.ndimage import gaussian_filter1d, grey_closing, grey_dilation

import telemetry_parser

# Filtering windows
MEDIAN_WINDOW_MS = 250.0 #[ms]
SAVGOL_WINDOW_MS = 500.0 #[ms]
SAVGOL_POLYORDER = 5

# =============================================================================
# UTILITIES
# =============================================================================

def sigmoid(x):
    """Logistic sigmoid, supporting scalars and NumPy arrays."""
    return 1.0 / (1.0 + np.exp(-x))


def odd_window(value, minimum=3, maximum=None):
    """Return the nearest valid odd integer window length."""
    n = max(minimum, int(value))
    if maximum is not None:
        maximum = max(minimum, int(maximum) | 1)
        n = min(n, maximum)
    return n | 1


# =============================================================================
# ANGULAR VELOCITY
# =============================================================================

def angular_velocity(q, fs):
    """Convert quaternion samples to angular velocity [rad/s]."""
    r = Rotation.from_quat(q[:, [1, 2, 3, 0]])
    dq = r[:-1].inv() * r[1:]
    rv = dq.as_rotvec()

    omega = np.zeros((len(q), 3), dtype=np.float64)
    omega[1:] = rv * fs
    return omega


# =============================================================================
# ADAPTIVE GYRO FILTER
# =============================================================================

def filter_gyro(omega, fs):
    """
    Detect noisy gyro regions and replace those regions with a strongly
    smoothed angular-velocity estimate. Clean regions remain unchanged.
    """
    def noise_probability(col):
        offset = 0 if np.count_nonzero(col[0::2]) > np.count_nonzero(col[1::2]) else 1
        p = np.abs(sigmoid(np.diff(col[offset::2])) - 0.5) * 2
        return np.interp(
            np.arange(len(col)),
            np.linspace(0, len(col) - 1, len(p)),
            p,
        )

    probability = np.apply_along_axis(noise_probability, 0, omega)

    # Bridge short gaps and expand detected noise regions.
    win_len = max(3, int(fs))
    p = grey_closing(probability, size=(win_len, 1))
    p = grey_dilation(p, size=(max(1, win_len // 4), 1))

    # Smooth the transition between raw and filtered data.
    p_smooth = np.clip(
        gaussian_filter1d(p, sigma=max(1.0, fs * 0.05), axis=0),
        0.0,
        1.0,
    )

    def smooth_axis(col):
        offset = 0 if np.count_nonzero(col[0::2]) > np.count_nonzero(col[1::2]) else 1
        x = col[offset::2]

        med_win = odd_window(fs * MEDIAN_WINDOW_MS / 2000, maximum=len(x))
        sg_win = odd_window(fs * SAVGOL_WINDOW_MS / 2000, minimum=SAVGOL_POLYORDER + 2, maximum=len(x))

        filtered = savgol_filter(
            medfilt(x, kernel_size=med_win),
            window_length=sg_win,
            polyorder=SAVGOL_POLYORDER,
        )

        return np.interp(
            np.arange(len(col)),
            np.linspace(0, len(col) - 1, len(filtered)),
            filtered,
        )

    omega_smooth = np.apply_along_axis(smooth_axis, 0, omega)

    return np.where(p > 0.3, omega_smooth, omega), p_smooth


# =============================================================================
# GYRO → QUATERNIONS
# =============================================================================

def integrate_gyro(omega, q0, fs):
    """Integrate angular velocity into quaternions."""
    dt = 1.0 / fs
    out = np.empty((len(omega), 4), dtype=np.float64)
    out[0] = q0 / np.linalg.norm(q0)

    for i in tqdm(range(1, len(omega)), desc="Integrating gyro", unit="sample"):
        v = omega[i] * dt
        angle = np.linalg.norm(v)

        if angle > 1e-12:
            s = np.sin(angle / 2.0) / angle
            dq = np.array([
                np.cos(angle / 2.0),
                v[0] * s,
                v[1] * s,
                v[2] * s,
            ])
        else:
            dq = np.array([1.0, v[0] / 2, v[1] / 2, v[2] / 2])

        out[i] = qmul(out[i - 1], dq)

    return out


# =============================================================================
# MP4 ATOM PARSER
# =============================================================================

def atoms(buf, start, end):
    """Iterate over MP4 atoms in [start, end)."""
    p = start

    while p + 8 <= end:
        size = int.from_bytes(buf[p:p + 4], "big")
        atom_type = buf[p + 4:p + 8]
        header = 8

        if size == 1:
            size = int.from_bytes(buf[p + 8:p + 16], "big")
            header = 16
        elif size == 0:
            size = end - p

        if size < header or p + size > end:
            break

        yield p, size, atom_type, p + header, p + size
        p += size


def child(buf, start, end, atom_type):
    return next(
        ((p, n, t_start, t_end)
         for p, n, t, t_start, t_end in atoms(buf, start, end)
         if t == atom_type),
        None,
    )


def djmd_samples(filename):
    """Return MP4 bytes and byte ranges for all djmd samples."""
    with open(filename, "rb") as f:
        buf = bytearray(f.read())

    moov = next(
        ((a, z) for _, _, t, a, z in atoms(buf, 0, len(buf)) if t == b"moov"),
        None,
    )

    if moov is None:
        raise RuntimeError("moov not found")

    ma, mz = moov

    for _, _, track_type, track_start, track_end in atoms(buf, ma, mz):
        if track_type != b"trak":
            continue

        mdia = child(buf, track_start, track_end, b"mdia")
        if not mdia:
            continue
        _, _, mdia_start, mdia_end = mdia

        minf = child(buf, mdia_start, mdia_end, b"minf")
        if not minf:
            continue
        _, _, minf_start, minf_end = minf

        stbl = child(buf, minf_start, minf_end, b"stbl")
        if not stbl:
            continue
        _, _, stbl_start, stbl_end = stbl

        stsd = child(buf, stbl_start, stbl_end, b"stsd")
        if not stsd:
            continue
        _, _, stsd_start, stsd_end = stsd

        p = stsd_start + 8
        found = False

        for _ in range(int.from_bytes(buf[stsd_start + 4:stsd_start + 8], "big")):
            size = int.from_bytes(buf[p:p + 4], "big")

            if buf[p + 4:p + 8] == b"djmd":
                found = True
                break

            p += size

        if not found:
            continue

        stsz = child(buf, stbl_start, stbl_end, b"stsz")
        stsc = child(buf, stbl_start, stbl_end, b"stsc")
        chunk_offsets = (
            child(buf, stbl_start, stbl_end, b"co64")
            or child(buf, stbl_start, stbl_end, b"stco")
        )

        if not stsz or not stsc or not chunk_offsets:
            raise RuntimeError("Incomplete djmd sample table")

        _, _, stsz_start, _ = stsz
        _, _, stsc_start, _ = stsc
        _, _, offsets_start, _ = chunk_offsets

        fixed_size = int.from_bytes(buf[stsz_start + 4:stsz_start + 8], "big")
        sample_count = int.from_bytes(buf[stsz_start + 8:stsz_start + 12], "big")

        sizes = (
            [fixed_size] * sample_count
            if fixed_size
            else [
                int.from_bytes(
                    buf[stsz_start + 12 + i * 4:stsz_start + 16 + i * 4],
                    "big",
                )
                for i in range(sample_count)
            ]
        )

        chunk_count = int.from_bytes(buf[offsets_start + 4:offsets_start + 8], "big")
        offset_step = 8 if buf[offsets_start - 4:offsets_start] == b"co64" else 4

        offsets = [
            int.from_bytes(
                buf[offsets_start + 8 + i * offset_step:
                    offsets_start + 8 + (i + 1) * offset_step],
                "big",
            )
            for i in range(chunk_count)
        ]

        entry_count = int.from_bytes(buf[stsc_start + 4:stsc_start + 8], "big")

        samples_per_chunk = [
            (
                int.from_bytes(
                    buf[stsc_start + 8 + i * 12:
                        stsc_start + 12 + i * 12],
                    "big",
                ),
                int.from_bytes(
                    buf[stsc_start + 12 + i * 12:
                        stsc_start + 16 + i * 12],
                    "big",
                ),
            )
            for i in range(entry_count)
        ]

        samples = []
        sample_index = 0

        for chunk_index, offset in enumerate(offsets, 1):
            count = next(
                count for first_chunk, count in reversed(samples_per_chunk)
                if first_chunk <= chunk_index
            )

            for _ in range(count):
                if sample_index >= sample_count:
                    break

                size = sizes[sample_index]
                samples.append((offset, size))
                offset += size
                sample_index += 1

        if len(samples) != sample_count:
            raise RuntimeError("djmd sample table mismatch")

        return buf, samples

    raise RuntimeError("djmd track not found")


# =============================================================================
# PROTOBUF
# =============================================================================

def varint(buf, pos, end):
    value = shift = 0

    while pos < end:
        byte = buf[pos]
        pos += 1
        value |= (byte & 0x7F) << shift

        if byte < 0x80:
            return value, pos

        shift += 7

    raise RuntimeError("Bad protobuf varint")


def pb_fields(buf, base=0):
    """Parse protobuf fields and return their locations/values."""
    fields = []
    pos, end = 0, len(buf)

    while pos < end:
        key, pos = varint(buf, pos, end)
        field, wire = key >> 3, key & 7

        if field == 0:
            break

        if wire == 0:
            _, pos = varint(buf, pos, end)

        elif wire == 1:
            if pos + 8 > end:
                break
            pos += 8

        elif wire == 2:
            size, pos = varint(buf, pos, end)
            if pos + size > end:
                break
            fields.append((field, wire, base + pos, base + pos + size, None))
            pos += size

        elif wire == 5:
            if pos + 4 > end:
                break

            fields.append((
                field,
                wire,
                base + pos,
                base + pos + 4,
                struct.unpack("<f", buf[pos:pos + 4])[0],
            ))
            pos += 4

        else:
            break

    return fields


def quaternion_messages(buf, base=0):
    """Recursively find plausible float32 quaternion messages."""
    try:
        fields = pb_fields(buf, base)
    except Exception:
        return []

    values, positions = {}, {}

    for field, wire, start, end, value in fields:
        if wire == 5 and field in (1, 2, 3, 4):
            values[field] = value
            positions[field] = (start, end)

    found = []

    if len(values) == 4:
        q = np.array(
            [values[1], values[2], values[3], values[4]],
            dtype=np.float64,
        )

        if np.all(np.isfinite(q)) and 0.85 < np.linalg.norm(q) < 1.15:
            found.append((q, [positions[i] for i in range(1, 5)]))

    for field, wire, start, end, _ in fields:
        if wire == 2:
            found.extend(
                quaternion_messages(buf[start - base:end - base], start)
            )

    return found


# =============================================================================
# QUATERNION CONVERSION
# =============================================================================

def qmul(a, b):
    """Quaternion multiplication for [w, x, y, z] convention."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b

    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


A = np.array([0.5, -0.5, -0.5, 0.5])
B = np.array([0.0, 0.0, 1.0, 0.0])
Ainv = A * np.array([1, -1, -1, -1])
Binv = B * np.array([1, -1, -1, -1])


def parser_to_raw(q):
    return qmul(qmul(Binv, q), Ainv)


def match_quaternions(candidates, targets):
    """Match telemetry quaternions to protobuf quaternion locations."""
    if len(candidates) != len(targets):
        raise RuntimeError(
            f"Quaternion count mismatch: protobuf={len(candidates)}, "
            f"telemetry={len(targets)}"
        )

    return [
        (
            positions,
            (lambda q: q / np.linalg.norm(q))(parser_to_raw(targets[i])),
        )
        for i, (_, positions) in enumerate(candidates)
    ]


# =============================================================================
# MAIN
# =============================================================================

src = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])

if not src:
    raise SystemExit("No file selected.")

dst = os.path.splitext(src)[0] + "_gyroFixed.mp4"

print("Reading DJI telemetry...")

parser = telemetry_parser.Parser(src)
raw = parser.telemetry()

print("Camera:", parser.camera, "| Model:", parser.model)

chunks = []

for record in tqdm(raw, desc="Finding quaternion chunks", unit="chunk"):
    try:
        if record["Quaternion"]["Data"]:
            chunks.append(record["Quaternion"]["Data"])
    except (KeyError, TypeError):
        pass

samples = [sample for chunk in chunks for sample in chunk]

print(f"Quaternion chunks: {len(chunks):,}")
print(f"Quaternion samples: {len(samples):,}")

q = np.array(
    [[s["v"]["w"], s["v"]["x"], s["v"]["y"], s["v"]["z"]] for s in samples],
    dtype=np.float64,
)

timestamps = np.array([s["t"] for s in samples], dtype=np.float64)

dt = np.median(np.diff(timestamps))
dt /= 1000.0
fs = 1.0 / dt

print(f"Gyro sample rate: {fs:.2f} Hz")

print("Calculating angular velocity...")
omega_raw = angular_velocity(q, fs)

print("Detecting gyro noise and filtering...")
omega_filtered, noise_probability = filter_gyro(omega_raw, fs)

print(
    f"Noise probability: {noise_probability.mean():.3f} mean / "
    f"{noise_probability.max():.3f} max"
)

print("Reconstructing quaternions...")
q2 = integrate_gyro(omega_filtered, q[0], fs)

print("Locating djmd samples...")
mp4, packets = djmd_samples(src)

if len(packets) != len(chunks):
    raise RuntimeError(
        f"djmd packets={len(packets):,}, "
        f"quaternion chunks={len(chunks):,}"
    )

print(f"Matched {len(packets):,} djmd packets")

k = 0

for ci, (offset, size) in enumerate(
    tqdm(packets, desc="Patching djmd", unit="packet")
):
    found = quaternion_messages(
        mp4[offset:offset + size],
        base=offset,
    )

    expected = len(chunks[ci])

    if len(found) != expected:
        raise RuntimeError(
            f"Packet {ci}: protobuf quaternion candidates={len(found)}, "
            f"telemetry quaternions={expected}"
        )

    for positions, new_q in match_quaternions(
        found,
        q2[k:k + expected],
    ):
        for (start, end), value in zip(positions, new_q):
            mp4[start:end] = struct.pack("<f", float(value))
        k += 1

if k != len(q2):
    raise RuntimeError(f"Patched {k:,}/{len(q2):,}")

print(f"Writing {dst}...")

with open(dst, "wb") as f:
    f.write(mp4)

if os.path.getsize(dst) != os.path.getsize(src):
    raise RuntimeError("Output size changed!")

print("\nDONE")
print("Input :", src)
print("Output:", dst)
print(f"Quaternions patched: {k:,}")
print(f"File size: {os.path.getsize(dst):,} bytes")