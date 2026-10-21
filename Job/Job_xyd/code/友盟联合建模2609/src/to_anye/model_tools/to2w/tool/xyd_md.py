
import math
import gzip
import base64
from typing import Optional

_0xc8fe = [0] * 512
_0x9859 = [0] * 256

for _i in range(8):
    _0xc8fe[_i] = 1 << _i
for _i in range(8, 256):
    _0xc8fe[_i] = _0xc8fe[_i - 4] ^ _0xc8fe[_i - 5] ^ _0xc8fe[_i - 6] ^ _0xc8fe[_i - 8]
for _i in range(255):
    _0x9859[_0xc8fe[_i]] = _i
for _i in range(255, 512):
    _0xc8fe[_i] = _0xc8fe[_i - 255]

def _0x5ab7(n):
    return _0xc8fe[n]

def _0xd72f(n):
    return _0x9859[n]

class _0xd66c:
    def __init__(self, num, shift):
        offset = 0
        for offset in range(len(num)):
            if num[offset] != 0:
                break
        self.num = num[offset:] + [0] * shift

    def __getitem__(self, index):
        return self.num[index]

    def __len__(self):
        return len(self.num)

    def __mul__(self, other):
        num = [0] * (len(self) + len(other) - 1)
        for i, item in enumerate(self):
            for j, other_item in enumerate(other):
                num[i + j] ^= _0x5ab7(_0xd72f(item) + _0xd72f(other_item))
        return _0xd66c(num, 0)

    def __mod__(self, other):
        difference = len(self) - len(other)
        if difference < 0:
            return self
        ratio = _0xd72f(self[0]) - _0xd72f(other[0])
        num = [
            item ^ _0x5ab7(_0xd72f(other_item) + ratio)
            for item, other_item in zip(self, other)
        ]
        if difference:
            num.extend(self[-difference:])
        return _0xd66c(num, 0) % other

_0x63b2 = (

    (1, 26, 19), (1, 26, 16), (1, 26, 13), (1, 26, 9),

    (1, 44, 34), (1, 44, 28), (1, 44, 22), (1, 44, 16),

    (1, 70, 55), (1, 70, 44), (2, 35, 17), (2, 35, 13),

    (1, 100, 80), (2, 50, 32), (2, 50, 24), (4, 25, 9),

    (1, 134, 108), (2, 67, 43), (2, 33, 15, 2, 34, 16), (2, 33, 11, 2, 34, 12),

    (2, 86, 68), (4, 43, 27), (4, 43, 19), (4, 43, 15),

    (2, 98, 78), (4, 49, 31), (2, 32, 14, 4, 33, 15), (4, 39, 13, 1, 40, 14),

    (2, 121, 97), (2, 60, 38, 2, 61, 39), (4, 40, 18, 2, 41, 19), (4, 40, 14, 2, 41, 15),

    (2, 146, 116), (3, 58, 36, 2, 59, 37), (4, 36, 16, 4, 37, 17), (4, 36, 12, 4, 37, 13),

    (2, 86, 68, 2, 87, 69), (4, 69, 43, 1, 70, 44), (6, 43, 19, 2, 44, 20), (6, 43, 15, 2, 44, 16),

    (4, 101, 81), (1, 80, 50, 4, 81, 51), (4, 50, 22, 4, 51, 23), (3, 36, 12, 8, 37, 13),

    (2, 116, 92, 2, 117, 93), (6, 58, 36, 2, 59, 37), (4, 46, 20, 6, 47, 21), (7, 42, 14, 4, 43, 15),

    (4, 133, 107), (8, 59, 37, 1, 60, 38), (8, 44, 20, 4, 45, 21), (12, 33, 11, 4, 34, 12),

    (3, 145, 115, 1, 146, 116), (4, 64, 40, 5, 65, 41), (11, 36, 16, 5, 37, 17), (11, 36, 12, 5, 37, 13),

    (5, 109, 87, 1, 110, 88), (5, 65, 41, 5, 66, 42), (5, 54, 24, 7, 55, 25), (11, 36, 12, 7, 37, 13),

    (5, 122, 98, 1, 123, 99), (7, 73, 45, 3, 74, 46), (15, 43, 19, 2, 44, 20), (3, 45, 15, 13, 46, 16),

    (1, 135, 107, 5, 136, 108), (10, 74, 46, 1, 75, 47), (1, 50, 22, 15, 51, 23), (2, 42, 14, 17, 43, 15),

    (5, 150, 120, 1, 151, 121), (9, 69, 43, 4, 70, 44), (17, 50, 22, 1, 51, 23), (2, 42, 14, 19, 43, 15),

    (3, 141, 113, 4, 142, 114), (3, 70, 44, 11, 71, 45), (17, 47, 21, 4, 48, 22), (9, 39, 13, 16, 40, 14),

    (3, 135, 107, 5, 136, 108), (3, 67, 41, 13, 68, 42), (15, 54, 24, 5, 55, 25), (15, 43, 15, 10, 44, 16),

    (4, 144, 116, 4, 145, 117), (17, 68, 42), (17, 50, 22, 6, 51, 23), (19, 46, 16, 6, 47, 17),

    (2, 139, 111, 7, 140, 112), (17, 74, 46), (7, 54, 24, 16, 55, 25), (34, 37, 13),

    (4, 151, 121, 5, 152, 122), (4, 75, 47, 14, 76, 48), (11, 54, 24, 14, 55, 25), (16, 45, 15, 14, 46, 16),

    (6, 147, 117, 4, 148, 118), (6, 73, 45, 14, 74, 46), (11, 54, 24, 16, 55, 25), (30, 46, 16, 2, 47, 17),

    (8, 132, 106, 4, 133, 107), (8, 75, 47, 13, 76, 48), (7, 54, 24, 22, 55, 25), (22, 45, 15, 13, 46, 16),

    (10, 142, 114, 2, 143, 115), (19, 74, 46, 4, 75, 47), (28, 50, 22, 6, 51, 23), (33, 46, 16, 4, 47, 17),

    (8, 152, 122, 4, 153, 123), (22, 73, 45, 3, 74, 46), (8, 53, 23, 26, 54, 24), (12, 45, 15, 28, 46, 16),

    (3, 147, 117, 10, 148, 118), (3, 73, 45, 23, 74, 46), (4, 54, 24, 31, 55, 25), (11, 45, 15, 31, 46, 16),

    (7, 146, 116, 7, 147, 117), (21, 73, 45, 7, 74, 46), (1, 53, 23, 37, 54, 24), (19, 45, 15, 26, 46, 16),

    (5, 145, 115, 10, 146, 116), (19, 75, 47, 10, 76, 48), (15, 54, 24, 25, 55, 25), (23, 45, 15, 25, 46, 16),

    (13, 145, 115, 3, 146, 116), (2, 74, 46, 29, 75, 47), (42, 54, 24, 1, 55, 25), (23, 45, 15, 28, 46, 16),

    (17, 145, 115), (10, 74, 46, 23, 75, 47), (10, 54, 24, 35, 55, 25), (19, 45, 15, 35, 46, 16),

    (17, 145, 115, 1, 146, 116), (14, 74, 46, 21, 75, 47), (29, 54, 24, 19, 55, 25), (11, 45, 15, 46, 46, 16),

    (13, 145, 115, 6, 146, 116), (14, 74, 46, 23, 75, 47), (44, 54, 24, 7, 55, 25), (59, 46, 16, 1, 47, 17),

    (12, 151, 121, 7, 152, 122), (12, 75, 47, 26, 76, 48), (39, 54, 24, 14, 55, 25), (22, 45, 15, 41, 46, 16),

    (6, 151, 121, 14, 152, 122), (6, 75, 47, 34, 76, 48), (46, 54, 24, 10, 55, 25), (2, 45, 15, 64, 46, 16),

    (17, 152, 122, 4, 153, 123), (29, 74, 46, 14, 75, 47), (49, 54, 24, 10, 55, 25), (24, 45, 15, 46, 46, 16),

    (4, 152, 122, 18, 153, 123), (13, 74, 46, 32, 75, 47), (48, 54, 24, 14, 55, 25), (42, 45, 15, 32, 46, 16),

    (20, 147, 117, 4, 148, 118), (40, 75, 47, 7, 76, 48), (43, 54, 24, 22, 55, 25), (10, 45, 15, 67, 46, 16),

    (19, 148, 118, 6, 149, 119), (18, 75, 47, 31, 76, 48), (34, 54, 24, 34, 55, 25), (20, 45, 15, 61, 46, 16),
)

_0x863e = {'L': 0, 'M': 1, 'Q': 2, 'H': 3}
_0xd3a4 = {'L': 1, 'M': 0, 'Q': 3, 'H': 2}

def _0x4224(version, error_correction):
    offset = _0x863e[error_correction]
    raw = _0x63b2[(version - 1) * 4 + offset]
    blocks = []
    for i in range(0, len(raw), 3):
        count, total_count, data_count = raw[i:i + 3]
        blocks.extend([(total_count, data_count)] * count)
    return blocks

_0xa04c = [
    [],
    [6, 18], [6, 22], [6, 26], [6, 30], [6, 34],
    [6, 22, 38], [6, 24, 42], [6, 26, 46], [6, 28, 50],
    [6, 30, 54], [6, 32, 58], [6, 34, 62],
    [6, 26, 46, 66], [6, 26, 48, 70], [6, 26, 50, 74],
    [6, 30, 54, 78], [6, 30, 56, 82], [6, 30, 58, 86],
    [6, 34, 62, 90],
    [6, 28, 50, 72, 94], [6, 26, 50, 74, 98], [6, 30, 54, 78, 102],
    [6, 28, 54, 80, 106], [6, 32, 58, 84, 110], [6, 30, 58, 86, 114],
    [6, 34, 62, 90, 118],
    [6, 26, 50, 74, 98, 122], [6, 30, 54, 78, 102, 126],
    [6, 26, 52, 78, 104, 130], [6, 30, 56, 82, 108, 134],
    [6, 34, 60, 86, 112, 138], [6, 30, 58, 86, 114, 142],
    [6, 34, 62, 90, 118, 146],
    [6, 30, 54, 78, 102, 126, 150], [6, 24, 50, 76, 102, 128, 154],
    [6, 28, 54, 80, 106, 132, 158], [6, 32, 58, 84, 110, 136, 162],
    [6, 26, 54, 82, 110, 138, 166], [6, 30, 58, 86, 114, 142, 170],
]

_0x26af = (1 << 10) | (1 << 8) | (1 << 5) | (1 << 4) | (1 << 2) | (1 << 1) | (1 << 0)
_0x4cba = (1 << 12) | (1 << 11) | (1 << 10) | (1 << 9) | (1 << 8) | (1 << 5) | (1 << 2) | (1 << 0)
_0x6b0f = (1 << 14) | (1 << 12) | (1 << 10) | (1 << 4) | (1 << 1)

def _0x592c(data):
    d = 0
    while data:
        d += 1
        data >>= 1
    return d

def _0x9323(data):
    d = data << 10
    while _0x592c(d) - _0x592c(_0x26af) >= 0:
        d ^= _0x26af << (_0x592c(d) - _0x592c(_0x26af))
    return ((data << 10) | d) ^ _0x6b0f

def _0x3d70(data):
    d = data << 12
    while _0x592c(d) - _0x592c(_0x4cba) >= 0:
        d ^= _0x4cba << (_0x592c(d) - _0x592c(_0x4cba))
    return (data << 12) | d

def _0x2ec6(pattern):
    if pattern == 0:
        return lambda i, j: (i + j) % 2 == 0
    if pattern == 1:
        return lambda i, j: i % 2 == 0
    if pattern == 2:
        return lambda i, j: j % 3 == 0
    if pattern == 3:
        return lambda i, j: (i + j) % 3 == 0
    if pattern == 4:
        return lambda i, j: (i // 2 + j // 3) % 2 == 0
    if pattern == 5:
        return lambda i, j: (i * j) % 2 + (i * j) % 3 == 0
    if pattern == 6:
        return lambda i, j: ((i * j) % 2 + (i * j) % 3) % 2 == 0
    if pattern == 7:
        return lambda i, j: ((i * j) % 3 + (i + j) % 2) % 2 == 0
    raise ValueError(f"Bad mask pattern: {pattern}")

class _0x5324:
    def __init__(self):
        self.buffer = []
        self.length = 0

    def __len__(self):
        return self.length

    def _0x837c(self, num, length):
        for i in range(length):
            self._0x7534(((num >> (length - i - 1)) & 1) == 1)

    def _0x7534(self, bit):
        buf_index = self.length // 8
        if len(self.buffer) <= buf_index:
            self.buffer.append(0)
        if bit:
            self.buffer[buf_index] |= 0x80 >> (self.length % 8)
        self.length += 1

_0x7e29 = 0xEC
_0x94fe = 0x11

def _0x3cb6(version, error_correction, data_bytes):
    buf = _0x5324()

    buf._0x837c(0b0100, 4)

    char_count_bits = 8 if version < 10 else 16
    buf._0x837c(len(data_bytes), char_count_bits)

    for b in data_bytes:
        buf._0x837c(b, 8)

    rs_blocks = _0x4224(version, error_correction)
    bit_limit = sum(b[1] * 8 for b in rs_blocks)

    if len(buf) > bit_limit:
        raise ValueError(
            f"数据溢出：需要 {len(buf)} bit，但版本 {version} / {error_correction} 仅容纳 {bit_limit} bit"
        )


    for _ in range(min(bit_limit - len(buf), 4)):
        buf._0x7534(False)

    delimit = len(buf) % 8
    if delimit:
        for _ in range(8 - delimit):
            buf._0x7534(False)

    bytes_to_fill = (bit_limit - len(buf)) // 8
    for i in range(bytes_to_fill):
        buf._0x837c(_0x7e29 if i % 2 == 0 else _0x94fe, 8)


    offset = 0
    max_dc = 0
    max_ec = 0
    dcdata = []
    ecdata = []

    for total_count, data_count in rs_blocks:
        ec_count = total_count - data_count
        max_dc = max(max_dc, data_count)
        max_ec = max(max_ec, ec_count)

        current_dc = [0xFF & buf.buffer[offset + i] for i in range(data_count)]
        offset += data_count


        rs_poly = _0xd66c([1], 0)
        for i in range(ec_count):
            rs_poly = rs_poly * _0xd66c([1, _0x5ab7(i)], 0)

        raw_poly = _0xd66c(current_dc, len(rs_poly) - 1)
        mod_poly = raw_poly % rs_poly
        mod_offset = len(mod_poly) - ec_count
        current_ec = [
            mod_poly[i + mod_offset] if (i + mod_offset) >= 0 else 0
            for i in range(ec_count)
        ]

        dcdata.append(current_dc)
        ecdata.append(current_ec)


    result = []
    for i in range(max_dc):
        for dc in dcdata:
            if i < len(dc):
                result.append(dc[i])
    for i in range(max_ec):
        for ec in ecdata:
            if i < len(ec):
                result.append(ec[i])
    return result

def _0xa0db(modules):
    n = len(modules)
    score = 0


    container = [0] * (n + 1)
    for row in range(n):
        this_row = modules[row]
        prev = this_row[0]
        length = 0
        for col in range(n):
            if this_row[col] == prev:
                length += 1
            else:
                if length >= 5:
                    container[length] += 1
                length = 1
                prev = this_row[col]
        if length >= 5:
            container[length] += 1
    for col in range(n):
        prev = modules[0][col]
        length = 0
        for row in range(n):
            if modules[row][col] == prev:
                length += 1
            else:
                if length >= 5:
                    container[length] += 1
                length = 1
                prev = modules[row][col]
        if length >= 5:
            container[length] += 1
    score += sum(container[l] * (l - 2) for l in range(5, n + 1))


    for row in range(n - 1):
        this_row = modules[row]
        next_row = modules[row + 1]
        col = 0
        while col < n - 1:
            tr = this_row[col + 1]
            if tr != next_row[col + 1]:
                col += 2
                continue
            if tr != this_row[col] or tr != next_row[col]:
                col += 1
                continue
            score += 3
            col += 1


    short_range = range(n - 10)
    for row in range(n):
        this_row = modules[row]
        col = 0
        while col < n - 10:
            if (
                not this_row[col + 1]
                and this_row[col + 4]
                and not this_row[col + 5]
                and this_row[col + 6]
                and not this_row[col + 9]
                and (
                    (this_row[col] and this_row[col + 2] and this_row[col + 3]
                     and not this_row[col + 7] and not this_row[col + 8] and not this_row[col + 10])
                    or
                    (not this_row[col] and not this_row[col + 2] and not this_row[col + 3]
                     and this_row[col + 7] and this_row[col + 8] and this_row[col + 10])
                )
            ):
                score += 40
            if this_row[col + 10]:
                col += 2
            else:
                col += 1
    for col in range(n):
        row = 0
        while row < n - 10:
            if (
                not modules[row + 1][col]
                and modules[row + 4][col]
                and not modules[row + 5][col]
                and modules[row + 6][col]
                and not modules[row + 9][col]
                and (
                    (modules[row][col] and modules[row + 2][col] and modules[row + 3][col]
                     and not modules[row + 7][col] and not modules[row + 8][col] and not modules[row + 10][col])
                    or
                    (not modules[row][col] and not modules[row + 2][col] and not modules[row + 3][col]
                     and modules[row + 7][col] and modules[row + 8][col] and modules[row + 10][col])
                )
            ):
                score += 40
            if modules[row + 10][col]:
                row += 2
            else:
                row += 1


    dark_count = sum(sum(r) for r in modules)
    percent = dark_count / (n * n)
    rating = int(abs(percent * 100 - 50) / 5)
    score += rating * 10

    return score

class _0x7143:

    def __init__(self, version=None, error_correction='L', box_size=10, border=4):
        self.version = version
        self.error_correction = error_correction.upper()
        self.box_size = box_size
        self.border = border
        self._data_bytes = b''
        self._modules = None
        self._modules_count = 0

    def _0x61e5(self, data):
        if isinstance(data, str):
            self._data_bytes = data.encode('utf-8')
        else:
            self._data_bytes = bytes(data)
        self._modules = None

    def _0x1e93(self, start=1):
        version = start
        while version <= 40:
            blocks = _0x4224(version, self.error_correction)
            bit_limit = sum(b[1] * 8 for b in blocks)
            char_count_bits = 8 if version < 10 else 16
            needed = 4 + char_count_bits + len(self._data_bytes) * 8
            if needed <= bit_limit:
                self.version = version
                return
            version += 1
        raise ValueError("数据过大，即使 Version 40 也无法容纳")

    def _0x9af7(self, row, col):
        for r in range(-1, 8):
            if row + r <= -1 or self._modules_count <= row + r:
                continue
            for c in range(-1, 8):
                if col + c <= -1 or self._modules_count <= col + c:
                    continue
                if (
                    (0 <= r <= 6 and c in (0, 6))
                    or (0 <= c <= 6 and r in (0, 6))
                    or (2 <= r <= 4 and 2 <= c <= 4)
                ):
                    self._modules[row + r][col + c] = True
                else:
                    self._modules[row + r][col + c] = False

    def _0x7e26(self):
        pos = _0xa04c[self.version - 1]
        for i in range(len(pos)):
            row = pos[i]
            for j in range(len(pos)):
                col = pos[j]
                if self._modules[row][col] is not None:
                    continue
                for r in range(-2, 3):
                    for c in range(-2, 3):
                        if r == -2 or r == 2 or c == -2 or c == 2 or (r == 0 and c == 0):
                            self._modules[row + r][col + c] = True
                        else:
                            self._modules[row + r][col + c] = False

    def _0xee9e(self):
        for r in range(8, self._modules_count - 8):
            if self._modules[r][6] is None:
                self._modules[r][6] = r % 2 == 0
        for c in range(8, self._modules_count - 8):
            if self._modules[6][c] is None:
                self._modules[6][c] = c % 2 == 0

    def _0xdc63(self, test, mask_pattern):
        data = (_0xd3a4[self.error_correction] << 3) | mask_pattern
        bits = _0x9323(data)


        for i in range(15):
            mod = (not test) and (((bits >> i) & 1) == 1)
            if i < 6:
                self._modules[i][8] = mod
            elif i < 8:
                self._modules[i + 1][8] = mod
            else:
                self._modules[self._modules_count - 15 + i][8] = mod


        for i in range(15):
            mod = (not test) and (((bits >> i) & 1) == 1)
            if i < 8:
                self._modules[8][self._modules_count - i - 1] = mod
            elif i < 9:
                self._modules[8][15 - i - 1 + 1] = mod
            else:
                self._modules[8][15 - i - 1] = mod


        self._modules[self._modules_count - 8][8] = not test

    def _0xc468(self, test):
        bits = _0x3d70(self.version)
        for i in range(18):
            mod = (not test) and (((bits >> i) & 1) == 1)
            self._modules[i // 3][i % 3 + self._modules_count - 8 - 3] = mod
        for i in range(18):
            mod = (not test) and (((bits >> i) & 1) == 1)
            self._modules[i % 3 + self._modules_count - 8 - 3][i // 3] = mod

    def _0x1c5c(self, data, mask_pattern):
        inc = -1
        row = self._modules_count - 1
        bit_index = 7
        byte_index = 0
        mask_fn = _0x2ec6(mask_pattern)
        data_len = len(data)

        for col in range(self._modules_count - 1, 0, -2):
            if col <= 6:
                col -= 1
            col_range = (col, col - 1)
            while True:
                for c in col_range:
                    if self._modules[row][c] is None:
                        dark = False
                        if byte_index < data_len:
                            dark = ((data[byte_index] >> bit_index) & 1) == 1
                        if mask_fn(row, c):
                            dark = not dark
                        self._modules[row][c] = dark
                        bit_index -= 1
                        if bit_index == -1:
                            byte_index += 1
                            bit_index = 7
                row += inc
                if row < 0 or self._modules_count <= row:
                    row -= inc
                    inc = -inc
                    break

    def _0x6f1e(self, test, mask_pattern):
        self._modules_count = self.version * 4 + 17
        self._modules = [[None] * self._modules_count for _ in range(self._modules_count)]


        self._0x9af7(0, 0)
        self._0x9af7(self._modules_count - 7, 0)
        self._0x9af7(0, self._modules_count - 7)
        self._0x7e26()
        self._0xee9e()
        self._0xdc63(test, mask_pattern)
        if self.version >= 7:
            self._0xc468(test)


        data_bytes = _0x3cb6(self.version, self.error_correction, self._data_bytes)
        self._0x1c5c(data_bytes, mask_pattern)

    def _0xfa06(self):
        if self.version is None:
            self._0x1e93()


        best_pattern = 0
        best_score = None
        for i in range(8):
            self._0x6f1e(test=True, mask_pattern=i)
            score = _0xa0db(self._modules)
            if best_score is None or score < best_score:
                best_score = score
                best_pattern = i


        self._0x6f1e(test=False, mask_pattern=best_pattern)
        return self._modules

    def _0xe9e3(self):
        if self._modules is None:
            self._0xfa06()
        if not self.border:
            return self._modules
        width = self._modules_count + self.border * 2
        code = [[False] * width for _ in range(self.border)]
        x_border = [False] * self.border
        for row in self._modules:
            code.append(x_border + list(row) + x_border)
        code.extend([[False] * width for _ in range(self.border)])
        return code

    def _0xacf0(self, box_size=None, border=None):
        if box_size is not None:
            self.box_size = box_size
        if border is not None:
            self.border = border
        matrix = self._0xe9e3()
        n = len(matrix)
        size = n * self.box_size
        img = [[0] * size for _ in range(size)]
        for r in range(n):
            for c in range(n):
                if matrix[r][c]:
                    for dr in range(self.box_size):
                        for dc in range(self.box_size):
                            img[r * self.box_size + dr][c * self.box_size + dc] = 1
        return img

    def _0xa577(self, box_size=None, border=None, fill_color='black', back_color='white'):
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("to_pil() 需要 Pillow 库，请先 pip install Pillow")
        if box_size is not None:
            self.box_size = box_size
        if border is not None:
            self.border = border
        matrix = self._0xe9e3()
        n = len(matrix)
        size = n * self.box_size
        img = Image.new('RGB', (size, size), back_color)
        pixels = img.load()
        for r in range(n):
            for c in range(n):
                if matrix[r][c]:
                    for dr in range(self.box_size):
                        for dc in range(self.box_size):
                            pixels[r * self.box_size + dr, c * self.box_size + dc] = fill_color \
                                if isinstance(fill_color, tuple) else _0x4a51(fill_color)
        return img

def _0x4a51(color):
    colors = {
        'black': (0, 0, 0),
        'white': (255, 255, 255),
        'red': (255, 0, 0),
        'green': (0, 128, 0),
        'blue': (0, 0, 255),
    }
    return colors.get(color.lower(), (0, 0, 0))

def _0xfa06(data, version=None, error_correction='L', box_size=10, border=4):
    qr = _0x7143(version=version, error_correction=error_correction,
                box_size=box_size, border=border)
    qr._0x61e5(data)
    return qr._0xacf0()

def _0x6e77(indices):
    if not indices:
        return ""
    indices = sorted(set(indices))
    deltas = [indices[0]] + [indices[i] - indices[i - 1] for i in range(1, len(indices))]

    def _0xfe23(value):
        result = []
        while value > 0x7F:
            result.append((value & 0x7F) | 0x80)
            value >>= 7
        result.append(value & 0x7F)
        return bytes(result)

    byte_data = b''.join(_0xfe23(d) for d in deltas)
    compressed = gzip.compress(byte_data)
    return base64.urlsafe_b64encode(compressed).decode('ascii').rstrip('=')

def _0xde2f(b64url_str):
    if not b64url_str:
        return []
    pad = '=' * (4 - len(b64url_str) % 4) if len(b64url_str) % 4 else ''
    try:
        compressed = base64.urlsafe_b64decode((b64url_str + pad).encode('ascii'))
        byte_data = gzip.decompress(compressed)
    except Exception as e:
        raise ValueError(f"解压失败: {e}")

    deltas = []
    i = 0
    while i < len(byte_data):
        value = 0
        shift = 0
        while True:
            byte = byte_data[i]
            i += 1
            value |= (byte & 0x7F) << shift
            if not (byte & 0x80):
                break
            shift += 7
        deltas.append(value)

    indices = [deltas[0]]
    for d in deltas[1:]:
        indices.append(indices[-1] + d)
    return indices

def _0x24f9(s, max_chunk_bytes=2900):
    return [s[i:i + max_chunk_bytes] for i in range(0, len(s), max_chunk_bytes)]

def _0xe903(indices, qr_version=40, error_correction='L',
                       box_size=4, border=2, max_chunk_bytes=2900):
    """
    将索引列表压缩后分片，每片生成一张二维码。

    参数:
        indices: 整数索引列表
        qr_version: 二维码版本 1-40
        error_correction: 纠错等级 'L'/'M'/'Q'/'H'
        box_size: 每个模块的像素倍数
        border: 四周静默区宽度（模块数）
        max_chunk_bytes: 每片最大字符数

    返回:
        (payloads, imgs)
        payloads: 分片文本列表（格式 P{i}/N:内容）
        imgs: 二维码图片列表（二维列表，可直接 plt.imshow）
    """
    b64url = _0x6e77(indices)
    chunks = _0x24f9(b64url, max_chunk_bytes)
    payloads = []
    imgs = []

    for i, chunk in enumerate(chunks):
        payload = f"P{i + 1}/{len(chunks)}:{chunk}"
        qr = _0x7143(version=qr_version, error_correction=error_correction,
                    box_size=box_size, border=border)
        qr._0x61e5(payload)
        img = qr._0xacf0()
        imgs.append(img)
        payloads.append(payload)

    print(f"压缩后 {len(b64url)} 字符 → {len(chunks)} 张二维码 "
          f"(V{qr_version}-{error_correction}, {len(imgs[0])}x{len(imgs[0])}px)")
    return payloads, imgs

def _0x89b8(scanned_parts):
    def _0xed39(s):
        if s.startswith("P") and ":" in s and "/" in s:
            return int(s[1:s.index("/")])
        return 999

    scanned_parts = sorted(scanned_parts, key=_0xed39)
    full_b64url = ""
    for s in scanned_parts:
        if ":" in s:
            full_b64url += s.split(":", 1)[1]
    return _0xde2f(full_b64url)

def _0x9d7e(images, max_cols=3, titles=None):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 未安装，跳过图片显示")
        for i, img in enumerate(images):
            print(f"  图片 {i + 1}: {len(img)}x{len(img[0])} 矩阵")
        return

    n = len(images)
    if n == 0:
        print("无图片可显示")
        return

    cols = min(max_cols, n)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    fig.suptitle(f"QR Code Parts (Total: {n})", fontsize=12)

    if n == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes] if cols > 1 else [[axes]]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    for idx in range(n):
        row = idx // cols
        col = idx % cols
        ax = axes[row][col]
        ax.imshow(images[idx], cmap='gray')
        title = titles[idx] if titles and idx < len(titles) else f'Part {idx + 1}/{n}'
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    for idx in range(n, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row][col].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
