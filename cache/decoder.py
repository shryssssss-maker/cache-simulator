# cache/decoder.py
# =============================================================================
# ADDRESS DECODER
# Converts raw memory addresses to cache tag/index/offset components
# =============================================================================

import math
from config import CONFIG


class AddressDecoder:
    """
    Splits a 32-bit memory address into tag, index, and offset.

    For our config (256B cache, 64B blocks, 4-way, 1 set):
        Offset bits = log2(64)  = 6  (which byte in block)
        Index bits  = log2(1)   = 0  (which set — only 1)
        Tag bits    = 32 - 6 - 0 = 26 (block identifier)

    Example:
        Address: 0x00001A3F (binary: 00000000000000000001101000111111)
        Offset:  6 LSBs    = 0b111111 = 63
        Index:   0 bits    = 0
        Tag:     26 MSBs   = 0b00000000000000000001101000 = 104
    """

    def __init__(self, cfg=CONFIG):
        self.block_size    = cfg['block_size']
        self.cache_size    = cfg['cache_size']
        self.associativity = cfg['associativity']
        self.address_bits  = cfg['address_bits']

        self.num_sets    = cfg['num_sets']
        self.offset_bits = cfg['offset_bits']
        self.index_bits  = cfg['index_bits']
        self.tag_bits    = cfg['tag_bits']

        # Bitmasks for extracting each field
        # Offset mask: 6 ones in LSB positions → 0b111111 = 0x3F
        self.offset_mask = (1 << self.offset_bits) - 1

        # Index mask: shifted above offset bits
        if self.index_bits > 0:
            self.index_mask = ((1 << self.index_bits) - 1) << self.offset_bits
        else:
            self.index_mask = 0

    def decode(self, address):
        """
        Decode a raw address into (tag, index, offset).

        Args:
            address: integer memory address (32-bit)

        Returns:
            (tag, index, offset) tuple of integers

        Example:
            decoder.decode(0x1A3F) → (tag=104, index=0, offset=63)
        """
        # Extract offset (bottom bits)
        offset = address & self.offset_mask

        # Extract index (middle bits)
        if self.index_bits > 0:
            index = (address & self.index_mask) >> self.offset_bits
        else:
            index = 0

        # Extract tag (top bits)
        tag = address >> (self.offset_bits + self.index_bits)

        return tag, index, offset

    def reconstruct(self, tag, index, offset):
        """
        Reconstruct original address from components.
        Used for validation — decode then reconstruct must equal original.

        Args:
            tag, index, offset: integer components

        Returns:
            reconstructed integer address
        """
        return (
            (tag << (self.offset_bits + self.index_bits)) |
            (index << self.offset_bits) |
            offset
        )

    def debug_address(self, address):
        """
        Print detailed breakdown of an address.
        Use this during development to verify decoder.
        """
        tag, index, offset = self.decode(address)
        reconstructed = self.reconstruct(tag, index, offset)

        print(f"\nAddress Breakdown:")
        print(f"  Raw address : {address:#010x} ({address})")
        print(f"  Tag         : {tag} ({tag:026b})")
        print(f"  Index       : {index} ({index:0{max(self.index_bits,1)}b})")
        print(f"  Offset      : {offset} ({offset:06b})")
        print(f"  Reconstruct : {reconstructed:#010x}")
        print(f"  Valid       : {'✅' if reconstructed == address else ' BUG!'}")

    def validate(self):
        """
        Run sanity checks on the decoder.
        Call this during initialization to catch config errors early.
        """
        test_addresses = [
            0x00000000,
            0xFFFFFFFF,
            0x00001A3F,
            0x00002B10,
            0x0000FFFF,
        ]

        all_passed = True
        for addr in test_addresses:
            tag, index, offset = self.decode(addr)

            # Reconstruct and verify
            reconstructed = self.reconstruct(tag, index, offset)
            if reconstructed != addr:
                print(f" DECODE FAILED for {addr:#010x}")
                all_passed = False
                continue

            # Verify ranges
            if offset >= self.block_size:
                print(f" OFFSET OUT OF RANGE: {offset} >= {self.block_size}")
                all_passed = False
                continue

            if index >= max(self.num_sets, 1):
                print(f"INDEX OUT OF RANGE: {index} >= {self.num_sets}")
                all_passed = False
                continue

        if all_passed:
            print("AddressDecoder validation:  All checks passed")
        return all_passed
