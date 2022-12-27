#!/usr/bin/env python3

""" Main module containing the code that parses/unpacks a FLIR file.

    The most important function here for end users is `unpack()` which starts
    the unpacking sequence. It returns a `FlyrThermogram`, defined in
    thermogram.py, so check there for the main way to interface with the
    results.

    Much of the documentation needed to implement this was gotten from
    exiftool (https://exiftool.org/TagNames/FLIR.html).
"""

# FIXME stop depending on PIL and nptyping

# Standard library imports
import os
import struct
from io import BytesIO, BufferedIOBase
from typing import Dict, List, Optional, BinaryIO, Tuple, Union

# Third party imports
import numpy as np
from nptyping import Array
from PIL import Image

# Self imports
from .thermogram import FlyrThermogram
from .palette_info import Palette
from .picture_in_picture import PictureInPicture


# Constants
segment_sep = b"\xff"
app1_marker = b"\xe1"
magic_flir_def = b"FLIR\x00"

chunk_app1_bytes_count = len(app1_marker)
chunk_length_bytes_count = 2
chunk_magic_bytes_count = len(magic_flir_def)
chunk_skip_bytes_count = 1
chunk_num_bytes_count = 1
chunk_tot_bytes_count = 1
chunk_partial_metadata_length = (
    chunk_app1_bytes_count + chunk_length_bytes_count + chunk_magic_bytes_count
)
chunk_metadata_length = (
    chunk_partial_metadata_length
    + chunk_skip_bytes_count
    + chunk_num_bytes_count
    + chunk_tot_bytes_count
)


def unpack(path_or_stream: Union[str, BinaryIO]) -> FlyrThermogram:
    """ Unpacks the FLIR image, meaning that it will return the thermal data
        embedded in the image.

        Parameters
        ----------
        path_or_stream : Union[str, BinaryIO]
            Either a path (string) to a FLIR file, or a byte stream such as
            BytesIO or file opened as `open(file_path, "rb")`.

        Returns
        -------
        FlyrThermogram
            When successful, a FlyrThermogram object containing thermogram data.
    """
    if isinstance(path_or_stream, str) and os.path.isfile(path_or_stream):
        with open(path_or_stream, "rb") as flirh:
            thermogram = unpack(flirh)
        thermogram.path = path_or_stream
        return thermogram
    elif isinstance(path_or_stream, BufferedIOBase):
        stream = path_or_stream
        flir_app1_stream = extract_flir_app1(stream)
        flir_records = parse_flir_app1(flir_app1_stream)
        thermogram = parse_thermal(flir_app1_stream, flir_records)

        return thermogram
    else:
        raise ValueError("Incorrect input")  # TODO improved error message


def extract_flir_app1(stream: BinaryIO) -> BinaryIO:
    """ Extracts the FLIR APP1 bytes.

        Parameters
        ---------
        stream : BinaryIO
            A full bytes stream of a JPEG file, expected to be a FLIR file.

        Raises
        ------
        ValueError
            When the file is invalid in one the next ways, a
            ValueError is thrown.

            * File is not a JPEG
            * A FLIR chunk number occurs more than once
            * The total chunks count is inconsistent over multiple chunks
            * No APP1 segments are successfully parsed

        Returns
        -------
        BinaryIO
            A bytes stream of the APP1 FLIR segments
    """
    # Check JPEG-ness
    magic_bytes = stream.read(2)
    if False:  # TODO check whether image is a correct jpeg
        raise ValueError("Invalid FLIR: given file is not a jpeg")

    chunks_count: Optional[int] = None
    chunks: Dict[int, bytes] = {}
    while True:
        b = stream.read(1)
        if b == b"":
            break

        if b != segment_sep:
            continue

        parsed_chunk = parse_flir_chunk(stream, chunks_count)
        if not parsed_chunk:
            continue

        chunks_count, chunk_num, chunk = parsed_chunk
        chunk_exists = chunks.get(chunk_num, None) is not None
        if chunk_exists:
            raise ValueError("Invalid FLIR: duplicate chunk number")
        chunks[chunk_num] = chunk

        # Encountered all chunks, break out of loop to process found metadata
        if chunk_num == chunks_count:
            break

    if chunks_count is None:
        raise ValueError("Invalid FLIR: no metadata encountered")

    flir_app1_bytes = b""
    for chunk_num in range(chunks_count + 1):
        flir_app1_bytes += chunks[chunk_num]

    flir_app1_stream = BytesIO(flir_app1_bytes)
    flir_app1_stream.seek(0)
    return flir_app1_stream


def parse_flir_chunk(
    stream: BinaryIO, chunks_count: Optional[int]
) -> Optional[Tuple[int, int, bytes]]:
    # Parse the chunk header. Headers are as follows (definition with example):
    #
    #     \xff\xe1<length: 2 bytes>FLIR\x00\x01<chunk nr: 1 byte><chunk count: 1 byte>
    #     \xff\xe1\xff\xfeFLIR\x00\x01\x01\x0b
    #
    # Meaning: Exif APP1, 65534 long, FLIR chunk 1 out of 12
    marker = stream.read(chunk_app1_bytes_count)

    length_bytes = stream.read(chunk_length_bytes_count)
    length = int.from_bytes(length_bytes, "big")
    length -= chunk_metadata_length
    magic_flir = stream.read(chunk_magic_bytes_count)

    if not (marker == app1_marker and magic_flir == magic_flir_def):
        # Seek back to just after byte b and continue searching for chunks
        stream.seek(-len(marker) - len(length_bytes) - len(magic_flir), 1)
        return None

    stream.seek(1, 1)  # skip 1 byte, unsure what it is for

    chunk_num = int.from_bytes(stream.read(chunk_num_bytes_count), "big")
    chunks_tot = int.from_bytes(stream.read(chunk_tot_bytes_count), "big")

    # Remember total chunks to verify metadata consistency
    if chunks_count is None:
        chunks_count = chunks_tot

    if (  # Check whether chunk metadata is consistent
        chunks_tot is None
        or chunk_num < 0
        or chunk_num > chunks_tot
        or chunks_tot != chunks_count
    ):
        raise ValueError(
            f"Invalid FLIR: inconsistent total chunks, should be 0 or greater, "
            "but is {chunks_tot}"
        )

    # FIXME There is  calculation error somewhere which causes length to be
    #       1 too small. So far it manually adding 1 seems sufficient to
    #       correct for it, but it's not the right way.
    #       Fix by figuring out what causes the off by 1 error.
    return chunks_tot, chunk_num, stream.read(length + 1)


def parse_thermal(
    stream: BinaryIO, records: Dict[int, Tuple[int, int, int, int]]
) -> FlyrThermogram:
    # FIXME Support formats other than subtype == PNG
    RECORD_IDX_RAW_DATA = 1
    RECORD_IDX_EMBEDDED_IMAGE = 14
    RECORD_IDX_CAMERA_INFO = 32
    RECORD_IDX_PALETTE_INFO = 34
    RECORD_IDX_PICTURE_IN_PICTURE_INFO = 42

    raw_data_md = records[RECORD_IDX_RAW_DATA]
    width, height, raw_data = parse_raw_data(stream, raw_data_md)

    embedded_image = None
    # try:
    #     embedded_image_md = records[RECORD_IDX_EMBEDDED_IMAGE]
    #     _, _, embedded_image = parse_embedded_image(stream, embedded_image_md)
    # except:
    #     pass

    camera_info_md = records[RECORD_IDX_CAMERA_INFO]
    camera_info = parse_camera_info(stream, camera_info_md)

    try:
        palette_info_md = records[RECORD_IDX_PALETTE_INFO]
        palette_info: Optional[Palette] = parse_palette_info(stream, palette_info_md)
    except:
        palette_info = None

    try:
        pip_info_md = records[RECORD_IDX_PICTURE_IN_PICTURE_INFO]
        pip_info: Optional[PictureInPicture] = parse_picture_in_picture_info(
            stream, pip_info_md
        )
    except:
        pip_info = None

    thermogram = FlyrThermogram(
        raw_data,
        camera_info,
        optical=embedded_image,
        palette=palette_info,
        picture_in_picture=pip_info,
    )
    return thermogram


def parse_flir_app1(stream: BinaryIO) -> Dict[int, Tuple[int, int, int, int]]:
    # 0x00 - string[4] file format ID = "FFF\0"
    # 0x04 - string[16] file creator: seen "\0","MTX IR\0","CAMCTRL\0"
    # 0x14 - int32u file format version = 100
    # 0x18 - int32u offset to record directory
    # 0x1c - int32u number of entries in record directory
    # 0x20 - int32u next free index ID = 2
    # 0x24 - int16u swap pattern = 0 (?)
    # 0x28 - int16u[7] spares
    # 0x34 - int32u[2] reserved
    # 0x3c - int32u checksum

    # 1. Read 0x40 bytes and verify that its contents equals AFF\0 or FFF\0
    file_format_id = stream.read(4)  # TODO the check

    # 2. Read FLIR record directory metadata (ref 3)
    stream.seek(16, 1)
    file_format_version = int.from_bytes(stream.read(4), "big")
    record_dir_offset = int.from_bytes(stream.read(4), "big")
    record_dir_entries_count = int.from_bytes(stream.read(4), "big")
    stream.seek(28, 1)
    checksum = int.from_bytes(stream.read(4), "big")

    # 3. Read record directory (which is a FLIR record entry repeated
    # `record_dir_entries_count` times)
    stream.seek(record_dir_offset)
    record_dir_stream = BytesIO(stream.read(32 * record_dir_entries_count))

    # First parse the record metadata
    record_details: Dict[int, Tuple[int, int, int, int]] = {}
    for record_nr in range(record_dir_entries_count):
        record_dir_stream.seek(0)
        details = parse_flir_record_metadata(stream, record_nr)
        if details:
            record_details[details[1]] = details

    # Then parse the actual records
    # for (entry_idx, type, offset, length) in record_details:
    #     parse_record = record_parsers[type]
    #     stream.seek(offset)
    #     record = BytesIO(stream.read(length + 36))  # + 36 needed to find end
    #     parse_record(record, offset, length)

    return record_details


def parse_flir_record_metadata(
    stream: BinaryIO, record_nr: int
) -> Optional[Tuple[int, int, int, int]]:
    # FLIR record entry (ref 3):
    # 0x00 - int16u record type
    # 0x02 - int16u record subtype: RawData 1=BE, 2=LE, 3=PNG; 1 for other record types
    # 0x04 - int32u record version: seen 0x64,0x66,0x67,0x68,0x6f,0x104
    # 0x08 - int32u index id = 1
    # 0x0c - int32u record offset from start of FLIR data
    # 0x10 - int32u record length
    # 0x14 - int32u parent = 0 (?)
    # 0x18 - int32u object number = 0 (?)
    # 0x1c - int32u checksum: 0 for no checksum
    entry = 32 * record_nr
    stream.seek(entry)
    record_type = int.from_bytes(stream.read(2), "big")
    if record_type < 1:
        return None

    # TODO convert unnecessary reads to seeks
    record_subtype = int.from_bytes(stream.read(2), "big")
    record_version = int.from_bytes(stream.read(4), "big")
    index_id = int.from_bytes(stream.read(4), "big")
    record_offset = int.from_bytes(stream.read(4), "big")
    record_length = int.from_bytes(stream.read(4), "big")
    parent = int.from_bytes(stream.read(4), "big")
    object_numer = int.from_bytes(stream.read(4), "big")
    checksum = int.from_bytes(stream.read(4), "big")

    return (entry, record_type, record_offset, record_length)


def parse_raw_data(
    stream: BinaryIO, metadata: Tuple[int, int, int, int]
) -> Tuple[int, int, Array[np.uint8, ..., ...]]:
    (_, _, offset, length) = metadata
    stream.seek(offset)

    # from PIL import ImageFile  # Uncomment to enable loading corrupt PNGs
    # ImageFile.LOAD_TRUNCATED_IMAGES = True

    stream.seek(2, 1)  # Skip first two bytes, TODO Explain the why of the skip
    width = int.from_bytes(stream.read(2), "little")
    height = int.from_bytes(stream.read(2), "little")

    # Read the bytes with the raw thermal data and decode using PIL
    length = min(width * height * 2, length)
    stream.seek(offset + 2 * 16)  # TODO document why 2 * 16
    thermal_bytes = stream.read(length)
    thermal_stream = BytesIO(thermal_bytes)

    # Some thermograms have the raw data stored as PNG files, others simply the
    # sequence of bytes
    if thermal_bytes[:4] != b"\x89PNG":
        u16s = []
        u16 = thermal_stream.read(2)
        while u16:
            u16s.append(int.from_bytes(u16, "little"))
            u16 = thermal_stream.read(2)
        thermal_np = np.array(u16s).reshape((height, width))
    else:
        # FLIR PNG data is in the wrong byte order
        thermal_img = Image.open(thermal_stream)
        thermal_np = np.array(thermal_img)
        fix_byte_order = np.vectorize(lambda x: (x >> 8) + ((x & 0x00FF) << 8))
        thermal_np = fix_byte_order(thermal_np)

    # Check shape
    if thermal_np.shape != (height, width):
        msg = "Invalid FLIR: metadata's width and height don't match thermal data's actual width and height ({} vs ({}, {})"
        msg = msg.format(thermal_np.shape, height, width)
        raise ValueError(msg)

    return width, height, thermal_np


def parse_embedded_image(
    stream: BinaryIO, metadata: Tuple[int, int, int, int]
) -> Tuple[int, int, Array[np.uint8, ..., ...]]:
    (entry_idx, _, offset, length) = metadata
    stream.seek(offset)

    # from PIL import ImageFile  # Uncomment to enable loading corrupt PNGs
    # ImageFile.LOAD_TRUNCATED_IMAGES = True

    stream.seek(2, 1)  # Skip first two bytes, TODO Explain the why of the skip
    width = int.from_bytes(stream.read(2), "little")
    height = int.from_bytes(stream.read(2), "little")

    stream.seek(offset + 2 * 16)  # TODO document why 2 * 16

    # Read the bytes with the embedded image and decode using PIL
    optical_bytes = stream.read(length)
    optical_stream = BytesIO(optical_bytes)
    optical_img = Image.open(optical_stream)
    optical_np = np.array(optical_img)

    # Check shape
    if optical_np.shape[:2] != (height, width):
        msg = "Invalid FLIR: metadata's width and height don't match optical data's actual width and height ({} vs ({}, {})"
        msg = msg.format(optical_np.shape, height, width)
        raise ValueError(msg)

    return width, height, optical_np


def parse_camera_info(
    stream: BinaryIO, metadata: Tuple[int, int, int, int]
) -> Dict[str, Union[int, float]]:
    (_, _, offset, _) = metadata
    stream.seek(offset + 32)

    emissivity = stream.read(4)
    object_distance = stream.read(4)
    refl_app_temp = stream.read(4)
    atmos_temp = stream.read(4)
    ir_window_temp = stream.read(4)
    ir_window_transm = stream.read(4)

    stream.seek(offset + 60)
    rel_humidity = stream.read(4)

    stream.seek(offset + 88)
    planck_r1 = stream.read(4)
    planck_b = stream.read(4)
    planck_f = stream.read(4)

    stream.seek(offset + 112)
    atmospheric_trans_alpha1 = stream.read(4)
    atmospheric_trans_alpha2 = stream.read(4)
    atmospheric_trans_beta1 = stream.read(4)
    atmospheric_trans_beta2 = stream.read(4)
    atmospheric_trans_x = stream.read(4)

    stream.seek(offset + 776)
    planck_o = stream.read(4)
    planck_r2 = stream.read(4)

    stream.seek(offset + 784)
    raw_value_range_min = stream.read(2)
    raw_value_range_max = stream.read(2)

    stream.seek(offset + 824)
    raw_value_median = stream.read(4)
    raw_value_range = stream.read(4)

    # Create result dictionary
    float_from_bytes = lambda v: struct.unpack("f", v)[0]
    camera_info = {
        "emissivity": float_from_bytes(emissivity),
        "object_distance": float_from_bytes(object_distance),
        "atmospheric_temperature": float_from_bytes(atmos_temp),
        "ir_window_temperature": float_from_bytes(ir_window_temp),
        "ir_window_transmission": float_from_bytes(ir_window_transm),
        "reflected_apparent_temperature": float_from_bytes(refl_app_temp),
        "relative_humidity": float_from_bytes(rel_humidity),
        "planck_r1": float_from_bytes(planck_r1),
        "planck_r2": float_from_bytes(planck_r2),
        "planck_b": float_from_bytes(planck_b),
        "planck_f": float_from_bytes(planck_f),
        "planck_o": struct.unpack("i", planck_o)[0],
        "atmospheric_trans_alpha1": float_from_bytes(atmospheric_trans_alpha1),
        "atmospheric_trans_alpha2": float_from_bytes(atmospheric_trans_alpha2),
        "atmospheric_trans_beta1": float_from_bytes(atmospheric_trans_beta1),
        "atmospheric_trans_beta2": float_from_bytes(atmospheric_trans_beta2),
        "atmospheric_trans_x": float_from_bytes(atmospheric_trans_x),
        "raw_value_range_min": struct.unpack("H", raw_value_range_min)[0],
        "raw_value_range_max": struct.unpack("H", raw_value_range_max)[0],
        "raw_value_median": struct.unpack("i", raw_value_median)[0],
        "raw_value_range": struct.unpack("i", raw_value_range)[0],
    }

    to_round = [
        "atmospheric_temperature",
        "ir_window_temperature",
        "reflected_apparent_temperature",
    ]
    camera_info = {k: round(v, 2) if k in to_round else v for k, v in camera_info.items()}

    return camera_info


def parse_palette_info(stream: BinaryIO, metadata: Tuple[int, int, int, int]) -> Palette:
    (_, _, offset, _) = metadata

    stream.seek(offset)
    num_colors = int.from_bytes(stream.read(6), "little")

    def read_color_tuple(stream):
        c1 = int.from_bytes(stream.read(1), "little")
        c2 = int.from_bytes(stream.read(1), "little")
        c3 = int.from_bytes(stream.read(1), "little")
        return (c1, c2, c3)

    above = read_color_tuple(stream)
    below = read_color_tuple(stream)
    overflow = read_color_tuple(stream)
    underflow = read_color_tuple(stream)
    isotherm1 = read_color_tuple(stream)
    isotherm2 = read_color_tuple(stream)
    stream.read(2)
    method = int.from_bytes(stream.read(1), "little")
    stretch = int.from_bytes(stream.read(1), "little")

    stream.seek(offset + 112)
    palette: bytes = stream.read(num_colors * 3)
    palette_ycc: List[Tuple[int, int, int]] = [
        (palette[i], palette[i + 1], palette[i + 2]) for i in range(0, len(palette), 3)
    ]

    return Palette(
        num_colors,
        above,
        below,
        overflow,
        underflow,
        isotherm1,
        isotherm2,
        method,
        stretch,
        "",  # TODO Read palette file name
        "",  # TODO Read palette name
        palette_ycc,
    )


def parse_picture_in_picture_info(
    stream: BinaryIO, metadata: Tuple[int, int, int, int]
) -> PictureInPicture:
    (_, _, offset, _) = metadata

    stream.seek(offset)

    ratio = stream.read(4)
    real_to_ir = struct.unpack("f", ratio)[0]
    offset_x = int.from_bytes(stream.read(2), "little", signed=True)
    offset_y = int.from_bytes(stream.read(2), "little", signed=True)
    pip_x1 = int.from_bytes(stream.read(2), "little", signed=True)
    pip_x2 = int.from_bytes(stream.read(2), "little", signed=True)
    pip_y1 = int.from_bytes(stream.read(2), "little", signed=True)
    pip_y2 = int.from_bytes(stream.read(2), "little", signed=True)

    return PictureInPicture(
        real_to_ir, offset_x, offset_y, pip_x1, pip_x2, pip_y1, pip_y2
    )


def main():
    import argparse
    import warnings
    import traceback as tb
    from pprint import pprint

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--camera-info", "-c", action="store_true", default=False, required=False
    )
    parser.add_argument("--palette-dest", default="", required=False)
    parser.add_argument("--render-dest", "-r", default="", required=False)
    parser.add_argument("--optical-dest", "-o", default="", required=False)
    parser.add_argument("--pnp-dest", "-p", default="", required=False)
    parser.add_argument("--emboss-dest", "-e", default="", required=False)
    parser.add_argument("thermogram")
    args = parser.parse_args()

    script = os.path.basename(__file__)
    if not os.path.isfile(args.thermogram):
        print(f"{script}: error: file {args.thermogram} doesn't exist")
        exit(1)

    try:
        thermogram = unpack(args.thermogram)

        if args.camera_info:
            pprint(thermogram.metadata)

        if args.palette_dest and thermogram.palette is not None:
            thermogram.palette.render_pil().save(args.palette_dest)
        elif args.palette_dest:
            warnings.warn(f"{script}: warning: no palette was extracted")

        if args.render_dest:
            thermogram.render_pil().save(args.render_dest)

        if args.optical_dest:
            thermogram.optical_pil.save(args.optical_dest)

        if args.pnp_dest:
            thermogram.picture_in_picture().save(args.pnp_dest)

        if args.emboss_dest:
            thermogram.render_pil(edge_emphasis=0.275).save(args.emboss_dest)
    except Exception as e:
        msg = f"{tb.format_exc()}\n{script}: error: failed unpacking {args.thermogram}"
        print(msg)


if __name__ == "__main__":
    main()
