import unittest
import sys
sys.path.insert(0, '..')
from cache.block import CacheBlock


class TestCacheBlock(unittest.TestCase):

    def test_initial_state_clean(self):
        block = CacheBlock(tag=1)
        self.assertFalse(block.dirty)

    def test_write_sets_dirty(self):
        block = CacheBlock(tag=1)
        block.write()
        self.assertTrue(block.dirty)

    def test_read_does_not_set_dirty(self):
        block = CacheBlock(tag=1)
        block.read()
        self.assertFalse(block.dirty)

    def test_frequency_increments_on_read(self):
        block = CacheBlock(tag=1)
        initial = block.frequency
        block.read()
        self.assertEqual(block.frequency, initial + 1)

    def test_frequency_increments_on_write(self):
        block = CacheBlock(tag=1)
        initial = block.frequency
        block.write()
        self.assertEqual(block.frequency, initial + 1)

    def test_recency_resets_on_read(self):
        block = CacheBlock(tag=1)
        block.tick(); block.tick(); block.tick()
        self.assertEqual(block.recency, 3)
        block.read()
        self.assertEqual(block.recency, 0)

    def test_recency_resets_on_write(self):
        block = CacheBlock(tag=1)
        block.tick(); block.tick()
        block.write()
        self.assertEqual(block.recency, 0)

    def test_tick_increments_age_and_recency(self):
        block = CacheBlock(tag=1)
        block.tick()
        self.assertEqual(block.age, 1)
        self.assertEqual(block.recency, 1)

    def test_to_state_normalized(self):
        block = CacheBlock(tag=1)
        state = block.to_state(max_val=1000)
        self.assertEqual(len(state), 4)
        for val in state:
            self.assertGreaterEqual(val, 0.0)
            self.assertLessEqual(val, 1.0)

    def test_dirty_in_state(self):
        block = CacheBlock(tag=1)
        self.assertEqual(block.to_state()[2], 0.0)  # clean
        block.write()
        self.assertEqual(block.to_state()[2], 1.0)  # dirty
