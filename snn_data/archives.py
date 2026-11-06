"""Read selected IC members without extracting source archives to disk."""
import ctypes as ct
import ctypes.util
import hashlib
from pathlib import PurePosixPath
import re
import zipfile

CONTRACT = re.compile(r'IC\d{2}(?:0[1-9]|1[0-2])$')
CONTINUOUS = frozenset(('ICL0', 'ICL1', 'ICL2', 'ICL3', 'ICL8', 'ICL9'))


def member_symbol(name):
    name = name.replace('\\', '/')
    if '中金所期货' not in name.split('/') or not name.lower().endswith('.csv'):
        return None
    symbol = PurePosixPath(name).stem.split('_')[0].upper()
    return symbol if CONTRACT.fullmatch(symbol) or symbol in CONTINUOUS else None


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def archive_format(path):
    with open(path, 'rb') as f:
        magic = f.read(6)
    if magic[:2] == b'PK':
        return 'zip'
    if magic == b'7z\xbc\xaf\x27\x1c':
        return '7z'
    raise ValueError(f'Unsupported archive signature: {path}')


class LibArchive:
    def __init__(self):
        library = ctypes.util.find_library('archive')
        if not library:
            raise RuntimeError('7z input requires the system libarchive library')
        self.lib = ct.CDLL(library)
        specs = [
            ('archive_read_new', ct.c_void_p, []),
            ('archive_read_support_filter_all', ct.c_int, [ct.c_void_p]),
            ('archive_read_support_format_all', ct.c_int, [ct.c_void_p]),
            ('archive_read_open_filename', ct.c_int, [ct.c_void_p, ct.c_char_p, ct.c_size_t]),
            ('archive_read_next_header', ct.c_int, [ct.c_void_p, ct.POINTER(ct.c_void_p)]),
            ('archive_entry_pathname_utf8', ct.c_char_p, [ct.c_void_p]),
            ('archive_entry_size', ct.c_int64, [ct.c_void_p]),
            ('archive_read_data', ct.c_ssize_t, [ct.c_void_p, ct.c_void_p, ct.c_size_t]),
            ('archive_read_free', ct.c_int, [ct.c_void_p]),
            ('archive_error_string', ct.c_char_p, [ct.c_void_p]),
        ]
        for name, result, args in specs:
            function = getattr(self.lib, name)
            function.restype, function.argtypes = result, args

    def read_members(self, path):
        lib = self.lib
        handle = lib.archive_read_new()
        lib.archive_read_support_filter_all(handle)
        lib.archive_read_support_format_all(handle)
        try:
            if lib.archive_read_open_filename(handle, str(path).encode(), 65536) < 0:
                raise RuntimeError(lib.archive_error_string(handle))
            entry = ct.c_void_p()
            buffer = ct.create_string_buffer(65536)
            while True:
                status = lib.archive_read_next_header(handle, ct.byref(entry))
                if status == 1:
                    break
                if status < 0:
                    raise RuntimeError(lib.archive_error_string(handle))
                name = lib.archive_entry_pathname_utf8(entry).decode('utf-8').replace('\\', '/')
                if member_symbol(name) is None:
                    continue
                size = lib.archive_entry_size(entry)
                parts = []
                while True:
                    length = lib.archive_read_data(handle, buffer, len(buffer))
                    if length < 0:
                        raise RuntimeError(lib.archive_error_string(handle))
                    if length == 0:
                        break
                    parts.append(buffer.raw[:length])
                payload = b''.join(parts)
                if len(payload) != size:
                    raise ValueError(f'Incomplete member: {path} / {name}')
                yield name, payload
        finally:
            lib.archive_read_free(handle)


def iter_ic_members(path, kind=None):
    kind = kind or archive_format(path)
    if kind == '7z':
        yield from LibArchive().read_members(path)
    elif kind == 'zip':
        with zipfile.ZipFile(path, metadata_encoding='gb18030') as archive:
            for info in archive.infolist():
                name = info.filename.replace('\\', '/')
                if not info.is_dir() and member_symbol(name):
                    yield name, archive.read(info)
    else:
        raise ValueError(f'Unknown archive format: {kind}')
