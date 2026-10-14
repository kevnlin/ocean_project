"""Reading selected levels of a chunked HDF5 variable from byte ranges; a small local file stands in for the remote one."""
import h5py
import numpy as np
import pytest

from ocean_tokenizer import hdf5_chunks as H

SHAPE, CHUNKS = (3, 13, 50, 70), (1, 7, 16, 32)


@pytest.fixture(params=[False, True], ids=["gzip", "gzip+shuffle"])
def stored(tmp_path, request):
    data = np.random.default_rng(0).normal(size=SHAPE).astype("float32")
    data[:, :, :5, :9] = 9.96921e36            # land
    path = tmp_path / "field.h5"
    with h5py.File(path, "w") as h:
        h.create_dataset("TEMP", data=data, chunks=CHUNKS, compression="gzip", compression_opts=1,
                         shuffle=request.param, fillvalue=np.float32(9.96921e36))
    return path, data, request.param


def local_reader(path):
    def read(offset, size):
        with open(path, "rb") as f:
            f.seek(offset)
            return f.read(size)
    return read


def test_index_lists_every_stored_chunk(stored):
    path, data, _ = stored
    with h5py.File(path, "r") as h:
        rows = H.chunk_index(h["TEMP"])
    assert rows.shape == (3 * 2 * 4 * 3, 6)
    assert set(map(tuple, rows[:, :4])) == {(t, k, y, x) for t in range(3) for k in (0, 7)
                                            for y in (0, 16, 32, 48) for x in (0, 32, 64)}
    assert (rows[:, 5] > 0).all()


def test_spans_cover_the_needed_depth_chunks_of_one_time_and_nothing_else(stored):
    path, data, _ = stored
    with h5py.File(path, "r") as h:
        rows = H.chunk_index(h["TEMP"])
    spans = H.spans(rows, CHUNKS, time_index=1, levels=[0, 3, 6])
    assert len(spans) == 1                                   # levels 0-6 are one depth chunk
    offset, size, members = spans[0]
    assert (members[:, 0] == 1).all() and (members[:, 1] == 0).all() and len(members) == 12
    assert offset == members[:, 4].min() and offset + size == (members[:, 4] + members[:, 5]).max()
    assert len(H.spans(rows, CHUNKS, time_index=1, levels=[0, 8])) == 2


@pytest.mark.parametrize("levels", [[0], [0, 3, 6], [2, 7, 12], list(range(13))])
def test_selected_levels_equal_a_direct_read(stored, levels):
    path, data, shuffle = stored
    with h5py.File(path, "r") as h:
        rows = H.chunk_index(h["TEMP"])
    for t in (0, 2):
        out = H.read_levels(local_reader(path), rows, SHAPE, CHUNKS, t, levels, np.dtype("float32"),
                            shuffle=shuffle, fill=np.float32(9.96921e36))
        assert out.shape == (len(levels), 50, 70) and out.dtype == np.float32
        assert np.array_equal(out, data[t, levels])


def test_a_chunk_that_was_never_written_reads_as_fill(stored):
    path, data, shuffle = stored
    with h5py.File(path, "r") as h:
        rows = H.chunk_index(h["TEMP"])
    keep = ~((rows[:, 0] == 0) & (rows[:, 2] == 16) & (rows[:, 3] == 32))
    out = H.read_levels(local_reader(path), rows[keep], SHAPE, CHUNKS, 0, [1, 9], np.dtype("float32"),
                        shuffle=shuffle, fill=np.float32(-1.))
    expected = data[0, [1, 9]].copy()
    expected[:, 16:32, 32:64] = -1.
    assert np.array_equal(out, expected)


def test_a_short_read_is_an_error_not_silent_garbage(stored):
    path, data, shuffle = stored
    with h5py.File(path, "r") as h:
        rows = H.chunk_index(h["TEMP"])
    with pytest.raises(ValueError, match="bytes"):
        H.read_levels(lambda offset, size: b"\0" * (size - 1), rows, SHAPE, CHUNKS, 0, [0],
                      np.dtype("float32"), shuffle=shuffle, fill=np.float32(0.))


def test_levels_must_exist_and_increase():
    rows = np.zeros((0, 6), dtype=np.int64)
    for bad in ([], [3, 1], [0, 0], [13], [-1]):
        with pytest.raises(ValueError, match="levels"):
            H.read_levels(lambda o, s: b"", rows, SHAPE, CHUNKS, 0, bad, np.dtype("float32"),
                          shuffle=False, fill=np.float32(0.))
