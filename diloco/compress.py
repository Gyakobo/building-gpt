"""
Pseudo-gradient compressors for DiLoCo's outer synchronization.

Each worker sends its pseudo-gradient delta_i = theta_global - theta_worker_i after H inner
steps. A compressor turns one tensor into a *payload*: a dict of the tensors that would
actually go over the wire. `decode` rebuilds a dense fp32 tensor from the payload, and
`payload_bytes` counts the bytes in it. Bytes are always counted from the real payload
tensors (bit-packed where values use fewer than 8 bits), never from a formula, so the
"bytes communicated" axis of every plot is exactly what the code would send.

Not counted: shapes, dtypes and the chunk layout, which are fixed for a given model and
known to every worker in advance.

Compressors (name -> what is sent):
    fp32          raw fp32 values                                        4 bytes/param
    fp16          values cast to fp16                                    2
    int8          symmetric int8, one fp32 scale per tensor              ~1
    int4          symmetric 4-bit, one fp16 scale per block of 128       0.516
    int2          4 levels {-3,-1,1,3}*s, one fp16 scale per 128-block   0.266
    sign          1 bit per value, fp16 mean |x| per 128-block           0.141
    topk<p>       per 4096-chunk top-k by |x|: fp32 value + int16 index  p% * 6
    randk<p>      per 4096-chunk random-k, mask from a shared seed: fp32 values only, p% * 4
    topk<p>_int4  top-k as above, values quantized to 4 bits with one fp16 scale per chunk

Error feedback (EF) lives in `ErrorFeedback`, outside the compressors, so every compressor
can be run with or without it.
"""
import math
import re

import torch

BLOCK = 128      # quantization block size (values per scale)
CHUNK = 4096     # top-k / random-k chunk size (indices fit in uint16)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _pad_rows(x, width):
    """Flatten x and zero-pad it to a multiple of `width`; returns (rows, n_real)."""
    flat = x.reshape(-1).float()
    n = flat.numel()
    pad = (-n) % width
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    return flat.view(-1, width), n


def _chunk_width(n):
    """Tensors smaller than CHUNK are one chunk of their own size."""
    return min(CHUNK, n)


def pack_bits(q, bits):
    """Pack integers in [0, 2**bits) (any shape) into a flat uint8 tensor."""
    per_byte = 8 // bits
    flat = q.reshape(-1).to(torch.uint8)
    pad = (-flat.numel()) % per_byte
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    flat = flat.view(-1, per_byte)
    out = torch.zeros(flat.size(0), dtype=torch.uint8, device=q.device)
    for j in range(per_byte):
        out |= flat[:, j] << (bits * j)
    return out


def unpack_bits(packed, bits, n):
    """Inverse of pack_bits: returns the first n values as int64."""
    per_byte = 8 // bits
    mask = (1 << bits) - 1
    cols = [(packed >> (bits * j)) & mask for j in range(per_byte)]
    return torch.stack(cols, dim=1).reshape(-1)[:n].long()


def payload_bytes(payload):
    return sum(t.numel() * t.element_size() for t in payload.values())


# ---------------------------------------------------------------------------
# compressors
# ---------------------------------------------------------------------------
class Compressor:
    name = "base"

    def encode(self, x, gen=None):
        raise NotImplementedError

    def decode(self, payload, like, gen=None):
        raise NotImplementedError

    def roundtrip(self, x, gen_seed=None):
        """Compress and decompress x. Returns (x_hat fp32 with x's shape, bytes sent).

        gen_seed seeds the random mask for random-k; the decoder rebuilds the same mask from
        the same seed, which is why no indices are sent."""
        enc_gen = dec_gen = None
        if gen_seed is not None:
            enc_gen = torch.Generator(device=x.device).manual_seed(gen_seed)
            dec_gen = torch.Generator(device=x.device).manual_seed(gen_seed)
        payload = self.encode(x, enc_gen)
        return self.decode(payload, x, dec_gen), payload_bytes(payload)


class FP32(Compressor):
    name = "fp32"

    def encode(self, x, gen=None):
        return {"v": x.float().clone()}

    def decode(self, p, like, gen=None):
        return p["v"].view_as(like)


class FP16(Compressor):
    name = "fp16"

    def encode(self, x, gen=None):
        return {"v": x.half()}

    def decode(self, p, like, gen=None):
        return p["v"].float().view_as(like)


class Int8(Compressor):
    """Symmetric int8 with one fp32 scale for the whole tensor."""
    name = "int8"

    def encode(self, x, gen=None):
        x = x.float()
        scale = x.abs().max().clamp_min(1e-30) / 127.0
        q = torch.round(x / scale).clamp_(-127, 127).to(torch.int8)
        return {"q": q, "scale": scale.reshape(1)}

    def decode(self, p, like, gen=None):
        return (p["q"].float() * p["scale"]).view_as(like)


class BlockQuant(Compressor):
    """Shared code for the block-wise low-bit quantizers (int4, int2, sign)."""
    bits = None

    def _codes(self, rows):        # -> (integer codes in [0, 2**bits), per-block scale)
        raise NotImplementedError

    def _values(self, codes, scale):
        raise NotImplementedError

    def encode(self, x, gen=None):
        rows, n = _pad_rows(x, BLOCK)
        codes, scale = self._codes(rows)
        return {"packed": pack_bits(codes, self.bits), "scale": scale.half()}

    def decode(self, p, like, gen=None):
        n = like.numel()
        n_rows = p["scale"].numel()
        codes = unpack_bits(p["packed"], self.bits, n_rows * BLOCK).view(n_rows, BLOCK)
        vals = self._values(codes, p["scale"].float().unsqueeze(1))
        return vals.reshape(-1)[:n].view_as(like)


class Int4(BlockQuant):
    """Symmetric 4-bit: values -7..7 times a per-block scale (code 15 is unused)."""
    name, bits = "int4", 4

    def _codes(self, rows):
        scale = rows.abs().amax(dim=1).clamp_min(1e-30) / 7.0
        q = torch.round(rows / scale.unsqueeze(1)).clamp_(-7, 7)
        return (q + 7).long(), scale

    def _values(self, codes, scale):
        return (codes.float() - 7.0) * scale


class Int2(BlockQuant):
    """4 symmetric levels {-3, -1, 1, 3} * s with s = block absmax / 3; nearest level."""
    name, bits = "int2", 2

    def _codes(self, rows):
        s = rows.abs().amax(dim=1).clamp_min(1e-30) / 3.0
        idx = torch.round((rows / s.unsqueeze(1) + 3.0) / 2.0).clamp_(0, 3)
        return idx.long(), s

    def _values(self, codes, scale):
        return (2.0 * codes.float() - 3.0) * scale


class Sign(BlockQuant):
    """1 bit per value: sign(x) * mean |x| of its block (sign of 0 is +)."""
    name, bits = "sign", 1

    def _codes(self, rows):
        return (rows >= 0).long(), rows.abs().mean(dim=1)

    def _values(self, codes, scale):
        return (2.0 * codes.float() - 1.0) * scale


class TopK(Compressor):
    """Per-chunk top-k by magnitude. Sends fp32 values + uint16 positions within the chunk.

    With quant_bits=4 the kept values are quantized like Int4, with one fp16 scale per chunk."""

    def __init__(self, density, quant_bits=None):
        assert 0 < density <= 1
        self.density, self.quant_bits = density, quant_bits
        pct = f"{density * 100:g}"
        self.name = f"topk{pct}" + (f"_int{quant_bits}" if quant_bits else "")

    def _k(self, width):
        return max(1, int(round(self.density * width)))

    def encode(self, x, gen=None):
        width = _chunk_width(x.numel())
        rows, n = _pad_rows(x, width)
        k = self._k(width)
        _, idx = rows.abs().topk(k, dim=1, sorted=False)
        vals = rows.gather(1, idx)
        payload = {"idx": idx.to(torch.int16)}   # 2 bytes; positions are < CHUNK = 4096
        if self.quant_bits == 4:
            scale = vals.abs().amax(dim=1).clamp_min(1e-30) / 7.0
            q = torch.round(vals / scale.unsqueeze(1)).clamp_(-7, 7) + 7
            payload["packed"] = pack_bits(q.long(), 4)
            payload["scale"] = scale.half()
        else:
            payload["v"] = vals.contiguous()
        return payload

    def decode(self, p, like, gen=None):
        n = like.numel()
        width = _chunk_width(n)
        n_rows = math.ceil(n / width)
        k = self._k(width)
        idx = p["idx"].long()
        if self.quant_bits == 4:
            q = unpack_bits(p["packed"], 4, n_rows * k).view(n_rows, k)
            vals = (q.float() - 7.0) * p["scale"].float().unsqueeze(1)
        else:
            vals = p["v"]
        out = torch.zeros(n_rows, width, device=like.device)
        out.scatter_(1, idx, vals)
        return out.reshape(-1)[:n].view_as(like)


class RandK(Compressor):
    """Per-chunk random-k, no rescaling (meant to be used with error feedback).

    The mask comes from a generator both sides seed identically, so only values are sent."""

    def __init__(self, density):
        self.density = density
        self.name = f"randk{density * 100:g}"

    def _mask_idx(self, n_rows, width, k, gen, device):
        scores = torch.rand(n_rows, width, generator=gen, device=device)
        return scores.topk(k, dim=1, sorted=False).indices

    def encode(self, x, gen=None):
        assert gen is not None, "random-k needs a seeded generator"
        width = _chunk_width(x.numel())
        rows, n = _pad_rows(x, width)
        k = max(1, int(round(self.density * width)))
        idx = self._mask_idx(rows.size(0), width, k, gen, x.device)
        return {"v": rows.gather(1, idx).contiguous()}

    def decode(self, p, like, gen=None):
        assert gen is not None, "random-k needs a seeded generator"
        n = like.numel()
        width = _chunk_width(n)
        n_rows = math.ceil(n / width)
        k = max(1, int(round(self.density * width)))
        idx = self._mask_idx(n_rows, width, k, gen, like.device)
        out = torch.zeros(n_rows, width, device=like.device)
        out.scatter_(1, idx, p["v"])
        return out.reshape(-1)[:n].view_as(like)


def make_compressor(name):
    """'fp32', 'fp16', 'int8', 'int4', 'int2', 'sign', 'topk1', 'topk0.1', 'randk1', 'topk1_int4'."""
    simple = {"fp32": FP32, "fp16": FP16, "int8": Int8, "int4": Int4, "int2": Int2, "sign": Sign}
    if name in simple:
        return simple[name]()
    m = re.fullmatch(r"(topk|randk)([0-9.]+)(_int4)?", name)
    if not m:
        raise ValueError(f"unknown compressor {name!r}")
    density = float(m.group(2)) / 100.0
    if m.group(1) == "randk":
        assert not m.group(3), "randk has no quantized variant"
        return RandK(density)
    return TopK(density, quant_bits=4 if m.group(3) else None)


# ---------------------------------------------------------------------------
# error feedback
# ---------------------------------------------------------------------------
class ErrorFeedback:
    """Worker-side error feedback, one fp32 buffer per parameter.

        m     = delta + beta * e        (beta = 1: classic EF; beta < 1: decayed, SparseLoCo-style)
        m_hat = decode(encode(m))       (what is actually sent)
        e     = m - m_hat               (what was left out, carried to the next round)
    """

    def __init__(self, params, beta=1.0):
        self.beta = beta
        self.buf = [torch.zeros_like(p, dtype=torch.float32) for p in params]

    def compress(self, compressor, deltas, seeds=None):
        out, nbytes = [], 0
        for j, (d, e) in enumerate(zip(deltas, self.buf)):
            m = d.float() + self.beta * e
            m_hat, b = compressor.roundtrip(m, None if seeds is None else seeds[j])
            e.copy_(m - m_hat)
            out.append(m_hat)
            nbytes += b
        return out, nbytes

    def norm(self):
        return math.sqrt(sum(float(e.pow(2).sum()) for e in self.buf))


def compress_all(compressor, deltas, seeds=None):
    """No error feedback: compress each tensor independently."""
    out, nbytes = [], 0
    for j, d in enumerate(deltas):
        d_hat, b = compressor.roundtrip(d, None if seeds is None else seeds[j])
        out.append(d_hat)
        nbytes += b
    return out, nbytes
