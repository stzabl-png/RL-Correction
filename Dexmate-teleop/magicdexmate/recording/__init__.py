from magicdexmate.recording.hand_data_writer import HandDataWriter

__all__ = ["HandDataWriter"]


TACTILE_KEYFRAME_DEFAULT = 30


def read_tactile(h5, name: str, index: int | None = None):
    """按文件记录的编码方式读回触觉图,返回原始像素。

    直接读 dataset 是不对的:开了差分录制的文件里存的是**帧间差**,而差分本身
    也是一张合法的 uint8 图,读错了不会报错,只会得到一批看起来像触觉图的噪声。
    所以读这两个 dataset 一律走这个函数。

        from magicdexmate.recording import read_tactile
        with h5py.File(p) as f:
            all_frames = read_tactile(f, "right_hand_tactile_raw")
            one_frame  = read_tactile(f, "right_hand_tactile_deform", index=k)

    index 给了就只解那一行:从它前面最近的关键帧往前累加,最多 keyframe-1 步。
    """
    import numpy as np
    enc = h5.attrs.get("tactile_encoding", "raw+lzf")
    if isinstance(enc, bytes):
        enc = enc.decode()
    ds = h5[name]
    plain = (not enc.startswith("delta")) or not name.endswith(("_deform", "_raw"))
    if plain:
        return np.asarray(ds[index]) if index is not None else np.asarray(ds)
    try:
        step = int(enc[len("delta"):].split("+")[0])
    except ValueError:
        step = TACTILE_KEYFRAME_DEFAULT
    if index is not None:
        k0 = (index // step) * step
        acc = np.asarray(ds[k0]).astype(np.int16)
        for i in range(k0 + 1, index + 1):
            acc = (acc + np.asarray(ds[i]).astype(np.int16)) % 256
        return acc.astype(np.uint8)
    data = np.asarray(ds)
    out = np.empty_like(data)
    acc = None
    for i in range(len(data)):
        if i % step == 0:
            acc = data[i].astype(np.int16)
        else:
            acc = (acc + data[i].astype(np.int16)) % 256
        out[i] = acc.astype(np.uint8)
    return out
