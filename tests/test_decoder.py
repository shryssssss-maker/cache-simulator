import unittest
import sys
sys.path.insert(0, '..')
from config import CONFIG
from cache.decoder import AddressDecoder


class TestAddressDecoder(unittest.TestCase):

    def setUp(self):
        self.decoder = AddressDecoder(CONFIG)

    def test_decode_reconstruct_zero(self):
        tag, idx, off = self.decoder.decode(0x00000000)
        self.assertEqual(self.decoder.reconstruct(tag, idx, off), 0x00000000)

    def test_decode_reconstruct_max(self):
        addr = (1 << CONFIG['address_bits']) - 1
        tag, idx, off = self.decoder.decode(addr)
        self.assertEqual(self.decoder.reconstruct(tag, idx, off), addr)

    def test_decode_reconstruct_typical(self):
        for addr in [0x00001A3F, 0x00002B10, 0x0000FFFF, 12345]:
            tag, idx, off = self.decoder.decode(addr)
            reconstructed = self.decoder.reconstruct(tag, idx, off)
            self.assertEqual(reconstructed, addr, f"Failed for {addr:#010x}")

    def test_offset_within_block_size(self):
        for addr in range(0, 10000, 64):
            _, _, offset = self.decoder.decode(addr)
            self.assertLess(offset, CONFIG['block_size'])

    def test_index_within_num_sets(self):
        num_sets = max(CONFIG['num_sets'], 1)
        for addr in range(1000):
            _, index, _ = self.decoder.decode(addr)
            self.assertLess(index, num_sets)

    def test_validate_passes(self):
        self.assertTrue(self.decoder.validate())
